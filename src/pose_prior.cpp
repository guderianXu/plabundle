#include <plabundle/constraints.h>
#include <plabundle/camera.h>

#include "pose_prior_internal.h"

#include <algorithm>
#include <cmath>

namespace plabundle
{
    namespace
    {
        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

        bool finitePose(const CameraPosePrior& prior)
        {
            FrameCamera camera;
            camera.cameraToWorldRotation = prior.cameraToWorldRotation;
            camera.cameraCenter = prior.cameraCenter;
            camera.focalXPixels = 1.0;
            camera.focalYPixels = 1.0;
            return validateFrameCamera(camera);
        }

        bool
        cholesky(const std::array<double, 36>& matrix, int dimension, std::array<double, 36>* lower, std::string* error)
        {
            lower->fill(0.0);
            double maximum = 0.0;
            for (int row = 0; row < dimension; ++row)
            {
                for (int column = 0; column < dimension; ++column)
                {
                    const double value = matrix[static_cast<std::size_t>(row * 6 + column)];
                    const double transpose = matrix[static_cast<std::size_t>(column * 6 + row)];
                    if (!std::isfinite(value) || !std::isfinite(transpose))
                    {
                        setError(error, "pose-prior uncertainty contains a non-finite value");
                        return false;
                    }
                    maximum = std::max(maximum, std::max(std::abs(value), std::abs(transpose)));
                    if (std::abs(value - transpose) > 1.0e-10 * std::max(1.0, maximum))
                    {
                        setError(error, "pose-prior covariance/information must be symmetric");
                        return false;
                    }
                }
            }
            const double threshold = std::max(1.0, maximum) * 1.0e-12;
            for (int row = 0; row < dimension; ++row)
            {
                for (int column = 0; column <= row; ++column)
                {
                    double value = matrix[static_cast<std::size_t>(row * 6 + column)];
                    for (int inner = 0; inner < column; ++inner)
                    {
                        value -= (*lower)[static_cast<std::size_t>(row * 6 + inner)] *
                                 (*lower)[static_cast<std::size_t>(column * 6 + inner)];
                    }
                    if (row == column)
                    {
                        if (!(value > threshold) || !std::isfinite(value))
                        {
                            setError(error, "pose-prior covariance/information must be positive definite");
                            return false;
                        }
                        (*lower)[static_cast<std::size_t>(row * 6 + column)] = std::sqrt(value);
                    }
                    else
                    {
                        (*lower)[static_cast<std::size_t>(row * 6 + column)] =
                            value / (*lower)[static_cast<std::size_t>(column * 6 + column)];
                    }
                }
            }
            return true;
        }

        std::array<double, 36> selectedMatrix(const CameraPosePrior& prior,
                                              const internal::PosePriorWhitening& whitening)
        {
            std::array<double, 36> selected{};
            for (int row = 0; row < whitening.residualSize; ++row)
            {
                const int source_row = whitening.componentIndices[static_cast<std::size_t>(row)];
                for (int column = 0; column < whitening.residualSize; ++column)
                {
                    const int source_column = whitening.componentIndices[static_cast<std::size_t>(column)];
                    selected[static_cast<std::size_t>(row * 6 + column)] =
                        prior.uncertaintyMatrix[static_cast<std::size_t>(source_row * 6 + source_column)];
                }
            }
            return selected;
        }

    } // namespace

    namespace internal
    {
        bool makePosePriorWhitening(const CameraPosePrior& prior, PosePriorWhitening* whitening, std::string* error)
        {
            if (!whitening)
            {
                setError(error, "pose-prior whitening output is null");
                return false;
            }
            *whitening = {};
            switch (prior.components)
            {
            case PosePriorComponents::Position:
                whitening->residualSize = 3;
                whitening->componentIndices = {{3, 4, 5, 0, 0, 0}};
                break;
            case PosePriorComponents::Rotation:
                whitening->residualSize = 3;
                whitening->componentIndices = {{0, 1, 2, 0, 0, 0}};
                break;
            case PosePriorComponents::RotationAndPosition:
                whitening->residualSize = 6;
                whitening->componentIndices = {{0, 1, 2, 3, 4, 5}};
                break;
            default:
                setError(error, "pose-prior component selection is invalid");
                return false;
            }

            if (prior.uncertainty == PosePriorUncertainty::IndependentSigmas)
            {
                const double rotation_sigma = prior.rotationSigmaDegrees * 3.14159265358979323846 / 180.0;
                if ((prior.components != PosePriorComponents::Position &&
                     (!std::isfinite(rotation_sigma) || rotation_sigma <= 0.0)) ||
                    (prior.components != PosePriorComponents::Rotation &&
                     (!std::isfinite(prior.positionSigmaMeters) || prior.positionSigmaMeters <= 0.0)))
                {
                    setError(error, "pose-prior scalar sigmas must be finite and positive");
                    return false;
                }
                for (int row = 0; row < whitening->residualSize; ++row)
                {
                    const int component = whitening->componentIndices[static_cast<std::size_t>(row)];
                    whitening->matrix[static_cast<std::size_t>(row * 6 + component)] =
                        component < 3 ? 1.0 / rotation_sigma : 1.0 / prior.positionSigmaMeters;
                }
                return true;
            }

            if (prior.uncertainty != PosePriorUncertainty::Covariance &&
                prior.uncertainty != PosePriorUncertainty::SqrtInformation)
            {
                setError(error, "pose-prior uncertainty representation is invalid");
                return false;
            }
            if (!std::all_of(prior.uncertaintyMatrix.begin(),
                             prior.uncertaintyMatrix.end(),
                             [](double value) { return std::isfinite(value); }))
            {
                setError(error, "pose-prior uncertainty contains a non-finite value");
                return false;
            }

            const int dimension = whitening->residualSize;
            const auto selected = selectedMatrix(prior, *whitening);
            if (prior.uncertainty == PosePriorUncertainty::Covariance)
            {
                std::array<double, 36> lower{};
                if (!cholesky(selected, dimension, &lower, error))
                {
                    return false;
                }
                std::array<double, 36> inverse_lower{};
                for (int column = 0; column < dimension; ++column)
                {
                    for (int row = 0; row < dimension; ++row)
                    {
                        double value = row == column ? 1.0 : 0.0;
                        for (int inner = 0; inner < row; ++inner)
                        {
                            value -= lower[static_cast<std::size_t>(row * 6 + inner)] *
                                     inverse_lower[static_cast<std::size_t>(inner * 6 + column)];
                        }
                        inverse_lower[static_cast<std::size_t>(row * 6 + column)] =
                            value / lower[static_cast<std::size_t>(row * 6 + row)];
                    }
                }
                for (int row = 0; row < dimension; ++row)
                {
                    for (int column = 0; column < dimension; ++column)
                    {
                        const int component = whitening->componentIndices[static_cast<std::size_t>(column)];
                        whitening->matrix[static_cast<std::size_t>(row * 6 + component)] =
                            inverse_lower[static_cast<std::size_t>(row * 6 + column)];
                    }
                }
                return true;
            }

            std::array<double, 36> information{};
            for (int row = 0; row < dimension; ++row)
            {
                for (int column = 0; column < dimension; ++column)
                {
                    for (int inner = 0; inner < dimension; ++inner)
                    {
                        information[static_cast<std::size_t>(row * 6 + column)] +=
                            selected[static_cast<std::size_t>(inner * 6 + row)] *
                            selected[static_cast<std::size_t>(inner * 6 + column)];
                    }
                }
            }
            std::array<double, 36> unused{};
            if (!cholesky(information, dimension, &unused, error))
            {
                return false;
            }
            for (int row = 0; row < dimension; ++row)
            {
                for (int column = 0; column < dimension; ++column)
                {
                    const int component = whitening->componentIndices[static_cast<std::size_t>(column)];
                    whitening->matrix[static_cast<std::size_t>(row * 6 + component)] =
                        selected[static_cast<std::size_t>(row * 6 + column)];
                }
            }
            return true;
        }
    } // namespace internal

    bool validateCameraPosePrior(const CameraPosePrior& prior, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        const int frame = static_cast<int>(prior.tangentFrame);
        if (!finitePose(prior) || frame < static_cast<int>(PosePriorTangentFrame::World) ||
            frame > static_cast<int>(PosePriorTangentFrame::PriorCamera))
        {
            setError(error, "pose-prior pose or tangent frame is invalid");
            return false;
        }
        internal::PosePriorWhitening whitening;
        return internal::makePosePriorWhitening(prior, &whitening, error);
    }

} // namespace plabundle
