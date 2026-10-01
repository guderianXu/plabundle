#include <plabundle/constraints.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/core/execution_policy.h>

#include "pose_prior_internal.h"

#include <algorithm>
#include <cmath>

namespace plabundle
{
    namespace
    {
        template <int Dimension> using FixedMatrix = plamatrix::Matrix<double, Dimension, Dimension>;

        template <int Dimension> FixedMatrix<Dimension> toMatrix(const std::array<double, 36>& values)
        {
            FixedMatrix<Dimension> matrix;
            for (int row = 0; row < Dimension; ++row)
            {
                for (int column = 0; column < Dimension; ++column)
                {
                    matrix(row, column) = values[static_cast<std::size_t>(row * 6 + column)];
                }
            }
            return matrix;
        }

        template <int Dimension> void copyToArray(const FixedMatrix<Dimension>& matrix, std::array<double, 36>* values)
        {
            values->fill(0.0);
            for (int row = 0; row < Dimension; ++row)
            {
                for (int column = 0; column < Dimension; ++column)
                {
                    (*values)[static_cast<std::size_t>(row * 6 + column)] = matrix(row, column);
                }
            }
        }

        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

        bool finitePose(const CameraPosePrior& prior)
        {
            return std::all_of(prior.cameraToWorldRotation.begin(),
                               prior.cameraToWorldRotation.end(),
                               [](double value) { return std::isfinite(value); }) &&
                   std::all_of(prior.cameraCenter.begin(),
                               prior.cameraCenter.end(),
                               [](double value) { return std::isfinite(value); });
        }

        template <int Dimension>
        bool choleskyFixed(const std::array<double, 36>& matrix, std::array<double, 36>* lower, std::string* error)
        {
            lower->fill(0.0);
            double maximum = 0.0;
            for (int row = 0; row < Dimension; ++row)
            {
                for (int column = 0; column < Dimension; ++column)
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
            const plamatrix::internal::ScopedExecutionPolicy cpu_only(plamatrix::internal::ExecutionPolicy::CpuOnly);
            const auto factor = toMatrix<Dimension>(matrix).llt();
            if (factor.info() != plamatrix::Success)
            {
                setError(error, "pose-prior covariance/information must be positive definite");
                return false;
            }
            const auto factor_lower = factor.matrixL();
            for (int diagonal = 0; diagonal < Dimension; ++diagonal)
            {
                const double pivot = factor_lower(diagonal, diagonal) * factor_lower(diagonal, diagonal);
                if (!(pivot > threshold) || !std::isfinite(pivot))
                {
                    setError(error, "pose-prior covariance/information must be positive definite");
                    return false;
                }
            }
            copyToArray(factor_lower, lower);
            return true;
        }

        bool
        cholesky(const std::array<double, 36>& matrix, int dimension, std::array<double, 36>* lower, std::string* error)
        {
            return dimension == 3 ? choleskyFixed<3>(matrix, lower, error) : choleskyFixed<6>(matrix, lower, error);
        }

        template <int Dimension> std::array<double, 36> inverseLower(const std::array<double, 36>& lower)
        {
            const plamatrix::internal::ScopedExecutionPolicy cpu_only(plamatrix::internal::ExecutionPolicy::CpuOnly);
            const auto inverse = toMatrix<Dimension>(lower).fullPivLu().solve(FixedMatrix<Dimension>::Identity());
            std::array<double, 36> result{};
            copyToArray(inverse, &result);
            return result;
        }

        template <int Dimension> std::array<double, 36> gramMatrix(const std::array<double, 36>& values)
        {
            const plamatrix::internal::ScopedExecutionPolicy cpu_only(plamatrix::internal::ExecutionPolicy::CpuOnly);
            const auto matrix = toMatrix<Dimension>(values);
            const auto gram = (matrix.transpose() * matrix).eval();
            std::array<double, 36> result{};
            copyToArray(gram, &result);
            return result;
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
                const std::array<double, 36> inverse_lower =
                    dimension == 3 ? inverseLower<3>(lower) : inverseLower<6>(lower);
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

            const std::array<double, 36> information =
                dimension == 3 ? gramMatrix<3>(selected) : gramMatrix<6>(selected);
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
