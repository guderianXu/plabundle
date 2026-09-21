#include <plabundle/backend.h>
#include <plabundle/options.h>

namespace plabundle
{

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
        bool enabled = false;
        switch (parameter)
        {
        case IntrinsicParameter::FocalLength:
            enabled = options.refineSharedFocalLength;
            break;
        case IntrinsicParameter::FocalAspectRatio:
            enabled = options.refineSharedFocalAspectRatio;
            break;
        case IntrinsicParameter::PrincipalPointX:
        case IntrinsicParameter::PrincipalPointY:
            enabled = options.refineSharedPrincipalPoint;
            break;
        case IntrinsicParameter::RadialK1:
            enabled = options.refineSharedRadialDistortion;
            break;
        case IntrinsicParameter::RadialK2:
        case IntrinsicParameter::RadialK3:
        case IntrinsicParameter::TangentialP1:
        case IntrinsicParameter::TangentialP2:
            enabled = options.refineSharedRadialDistortion && options.refineSharedHighOrderDistortion;
            break;
        case IntrinsicParameter::Count:
            return false;
        }

        const std::size_t index = static_cast<std::size_t>(parameter);
        return enabled && (!options.useSharedIntrinsicParameterMask || options.sharedIntrinsicParameterMask[index]);
    }

} // namespace plabundle
