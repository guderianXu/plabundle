#include <plabundle/backend.h>
#include <plabundle/options.h>

namespace plabundle
{
    namespace
    {
        template <typename Calibration>
        bool parameterEnabled(const Calibration& calibration, IntrinsicParameter parameter) noexcept
        {
            bool enabled = false;
            switch (parameter)
            {
            case IntrinsicParameter::FocalLength:
                enabled = calibration.refineSharedFocalLength;
                break;
            case IntrinsicParameter::FocalAspectRatio:
                enabled = calibration.refineSharedFocalAspectRatio;
                break;
            case IntrinsicParameter::PrincipalPointX:
            case IntrinsicParameter::PrincipalPointY:
                enabled = calibration.refineSharedPrincipalPoint;
                break;
            case IntrinsicParameter::RadialK1:
                enabled = calibration.refineSharedModelCoefficients || calibration.refineSharedRadialDistortion;
                break;
            case IntrinsicParameter::RadialK2:
            case IntrinsicParameter::RadialK3:
            case IntrinsicParameter::TangentialP1:
            case IntrinsicParameter::TangentialP2:
                enabled = (calibration.refineSharedModelCoefficients || calibration.refineSharedRadialDistortion) &&
                          calibration.refineSharedHighOrderDistortion;
                break;
            case IntrinsicParameter::SkewB2:
            case IntrinsicParameter::RadialK4:
            case IntrinsicParameter::TangentialP3:
            case IntrinsicParameter::TangentialP4:
                enabled = calibration.refineSharedMetashapeParameters;
                break;
            case IntrinsicParameter::Count:
                return false;
            }

            const std::size_t index = static_cast<std::size_t>(parameter);
            return enabled &&
                   (!calibration.useSharedIntrinsicParameterMask || calibration.sharedIntrinsicParameterMask[index]);
        }
    } // namespace

    const char* backendName(Backend backend) noexcept
    {
        switch (backend)
        {
        case Backend::Auto:
            return "auto";
        case Backend::PlaMatrixCpu:
            return "plamatrix_cpu";
        case Backend::PlaMatrixCuda:
            return "plamatrix_cuda";
        case Backend::PlaMatrixVulkan:
            return "plamatrix_vulkan";
        case Backend::PlaMatrixOpenCl:
            return "plamatrix_opencl";
        }
        return "unknown";
    }

    const char* solveStatusName(SolveStatus status) noexcept
    {
        switch (status)
        {
        case SolveStatus::NotRun:
            return "not_run";
        case SolveStatus::Success:
            return "success";
        case SolveStatus::NoConvergence:
            return "no_convergence";
        case SolveStatus::Cancelled:
            return "cancelled";
        case SolveStatus::InvalidInput:
            return "invalid_input";
        case SolveStatus::UnsupportedConfiguration:
            return "unsupported_configuration";
        case SolveStatus::BackendUnavailable:
            return "backend_unavailable";
        case SolveStatus::NumericalFailure:
            return "numerical_failure";
        }
        return "unknown";
    }

    bool sharedIntrinsicParameterEnabled(const Options& options, IntrinsicParameter parameter) noexcept
    {
        return parameterEnabled(options, parameter);
    }

    bool sharedIntrinsicParameterEnabled(const SolveOptions& options, IntrinsicParameter parameter) noexcept
    {
        return parameterEnabled(options.calibration, parameter);
    }

} // namespace plabundle
