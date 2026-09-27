#include <plabundle/constraints.h>

#include "control_point_internal.h"

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

        bool cholesky(const std::array<double, 9>& matrix, std::array<double, 9>* lower, std::string* error)
        {
            lower->fill(0.0);
            double maximum = 0.0;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const double value = matrix[static_cast<std::size_t>(row * 3 + column)];
                    const double transpose = matrix[static_cast<std::size_t>(column * 3 + row)];
                    if (!std::isfinite(value) || !std::isfinite(transpose))
                    {
                        setError(error, "control-point covariance/information contains a non-finite value");
                        return false;
                    }
                    maximum = std::max(maximum, std::max(std::abs(value), std::abs(transpose)));
                    if (std::abs(value - transpose) > 1.0e-10 * std::max(1.0, maximum))
                    {
                        setError(error, "control-point covariance/information must be symmetric");
                        return false;
                    }
                }
            }

            const double threshold = std::max(1.0, maximum) * 1.0e-12;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column <= row; ++column)
                {
                    double value = matrix[static_cast<std::size_t>(row * 3 + column)];
                    for (int inner = 0; inner < column; ++inner)
                    {
                        value -= (*lower)[static_cast<std::size_t>(row * 3 + inner)] *
                                 (*lower)[static_cast<std::size_t>(column * 3 + inner)];
                    }
                    if (row == column)
                    {
                        if (!(value > threshold) || !std::isfinite(value))
                        {
                            setError(error, "control-point covariance/information must be positive definite");
                            return false;
                        }
                        (*lower)[static_cast<std::size_t>(row * 3 + column)] = std::sqrt(value);
                    }
                    else
                    {
                        (*lower)[static_cast<std::size_t>(row * 3 + column)] =
                            value / (*lower)[static_cast<std::size_t>(column * 3 + column)];
                    }
                }
            }
            return true;
        }

    } // namespace

    namespace internal
    {
        bool makeControlPointWhitening(const ControlPointConstraint& constraint,
                                       ControlPointWhitening* whitening,
                                       std::string* error)
        {
            if (!whitening)
            {
                setError(error, "control-point whitening output is null");
                return false;
            }
            *whitening = {};
            switch (constraint.uncertainty)
            {
            case ControlPointUncertainty::IsotropicSigma:
                if (!std::isfinite(constraint.sigmaMeters) || constraint.sigmaMeters <= 0.0)
                {
                    setError(error, "control-point sigma must be finite and positive");
                    return false;
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    whitening->matrix[static_cast<std::size_t>(axis * 3 + axis)] = 1.0 / constraint.sigmaMeters;
                }
                return true;
            case ControlPointUncertainty::Covariance:
            {
                std::array<double, 9> lower{};
                if (!cholesky(constraint.uncertaintyMatrix, &lower, error))
                {
                    return false;
                }
                for (int column = 0; column < 3; ++column)
                {
                    for (int row = 0; row < 3; ++row)
                    {
                        double value = row == column ? 1.0 : 0.0;
                        for (int inner = 0; inner < row; ++inner)
                        {
                            value -= lower[static_cast<std::size_t>(row * 3 + inner)] *
                                     whitening->matrix[static_cast<std::size_t>(inner * 3 + column)];
                        }
                        whitening->matrix[static_cast<std::size_t>(row * 3 + column)] =
                            value / lower[static_cast<std::size_t>(row * 3 + row)];
                    }
                }
                return true;
            }
            case ControlPointUncertainty::SqrtInformation:
            {
                if (!std::all_of(constraint.uncertaintyMatrix.begin(),
                                 constraint.uncertaintyMatrix.end(),
                                 [](double value) { return std::isfinite(value); }))
                {
                    setError(error, "control-point square-root information contains a non-finite value");
                    return false;
                }
                std::array<double, 9> information{};
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        for (int inner = 0; inner < 3; ++inner)
                        {
                            information[static_cast<std::size_t>(row * 3 + column)] +=
                                constraint.uncertaintyMatrix[static_cast<std::size_t>(inner * 3 + row)] *
                                constraint.uncertaintyMatrix[static_cast<std::size_t>(inner * 3 + column)];
                        }
                    }
                }
                std::array<double, 9> unused{};
                if (!cholesky(information, &unused, error))
                {
                    return false;
                }
                whitening->matrix = constraint.uncertaintyMatrix;
                return true;
            }
            }
            setError(error, "control-point uncertainty representation is invalid");
            return false;
        }

        bool controlPointRmsUncertaintyMeters(const ControlPointConstraint& constraint,
                                              double* rms_uncertainty,
                                              std::string* error)
        {
            if (!rms_uncertainty)
            {
                setError(error, "control-point RMS uncertainty output is null");
                return false;
            }
            ControlPointWhitening whitening;
            if (!makeControlPointWhitening(constraint, &whitening, error))
            {
                return false;
            }
            const auto& matrix = whitening.matrix;
            const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                                       matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                                       matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
            if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-18)
            {
                setError(error, "control-point whitening matrix is singular");
                return false;
            }
            const double inverse_scale = 1.0 / determinant;
            const std::array<double, 9> inverse{{
                (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_scale,
                (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_scale,
                (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_scale,
                (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse_scale,
                (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_scale,
                (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_scale,
                (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse_scale,
                (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse_scale,
                (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_scale,
            }};
            double squared_uncertainty = 0.0;
            for (const double value : inverse)
            {
                squared_uncertainty += value * value;
            }
            *rms_uncertainty = std::sqrt(squared_uncertainty);
            return std::isfinite(*rms_uncertainty) && *rms_uncertainty > 0.0;
        }
    } // namespace internal

    bool validateControlPointConstraint(const ControlPointConstraint& constraint, std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!std::all_of(
                constraint.point.begin(), constraint.point.end(), [](double value) { return std::isfinite(value); }) ||
            !std::isfinite(constraint.weight) || constraint.weight <= 0.0)
        {
            setError(error, "control-point target and weight must be finite and weight must be positive");
            return false;
        }
        internal::ControlPointWhitening whitening;
        return internal::makeControlPointWhitening(constraint, &whitening, error);
    }

} // namespace plabundle
