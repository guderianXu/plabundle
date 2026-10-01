#include <plabundle/constraints.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/core/execution_policy.h>

#include "control_point_internal.h"

#include <algorithm>
#include <cmath>

namespace plabundle
{
    namespace
    {
        plamatrix::Matrix3d toMatrix(const std::array<double, 9>& values)
        {
            plamatrix::Matrix3d matrix;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    matrix(row, column) = values[static_cast<std::size_t>(row * 3 + column)];
                }
            }
            return matrix;
        }

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
            const plamatrix::internal::ScopedExecutionPolicy cpu_only(plamatrix::internal::ExecutionPolicy::CpuOnly);
            const auto factor = toMatrix(matrix).llt();
            if (factor.info() != plamatrix::Success)
            {
                setError(error, "control-point covariance/information must be positive definite");
                return false;
            }
            const auto factor_lower = factor.matrixL();
            for (int diagonal = 0; diagonal < 3; ++diagonal)
            {
                const double pivot = factor_lower(diagonal, diagonal) * factor_lower(diagonal, diagonal);
                if (!(pivot > threshold) || !std::isfinite(pivot))
                {
                    setError(error, "control-point covariance/information must be positive definite");
                    return false;
                }
            }
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column <= row; ++column)
                {
                    (*lower)[static_cast<std::size_t>(row * 3 + column)] = factor_lower(row, column);
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
                const plamatrix::internal::ScopedExecutionPolicy cpu_only(
                    plamatrix::internal::ExecutionPolicy::CpuOnly);
                const auto inverse = toMatrix(lower).fullPivLu().solve(plamatrix::Matrix3d::Identity());
                for (int column = 0; column < 3; ++column)
                {
                    for (int row = 0; row < 3; ++row)
                    {
                        whitening->matrix[static_cast<std::size_t>(row * 3 + column)] = inverse(row, column);
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
                const plamatrix::internal::ScopedExecutionPolicy cpu_only(
                    plamatrix::internal::ExecutionPolicy::CpuOnly);
                const auto weight = toMatrix(constraint.uncertaintyMatrix);
                const auto information_matrix = (weight.transpose() * weight).eval();
                std::array<double, 9> information{};
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        information[static_cast<std::size_t>(row * 3 + column)] = information_matrix(row, column);
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
            const plamatrix::internal::ScopedExecutionPolicy cpu_only(plamatrix::internal::ExecutionPolicy::CpuOnly);
            const auto factor = toMatrix(matrix).fullPivLu();
            const double determinant = factor.determinant();
            if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-18)
            {
                setError(error, "control-point whitening matrix is singular");
                return false;
            }
            const auto inverse = factor.solve(plamatrix::Matrix3d::Identity());
            double squared_uncertainty = 0.0;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const double value = inverse(row, column);
                    squared_uncertainty += value * value;
                }
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
