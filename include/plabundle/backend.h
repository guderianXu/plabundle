#pragma once

#include <string>

namespace plabundle
{

    enum class Backend
    {
        Auto,
        PlaMatrixCpu,
        PlaMatrixCuda,
        PlaMatrixVulkan,
        PlaMatrixOpenCl,
    };

    enum class SolveStatus
    {
        NotRun,
        Success,
        NoConvergence,
        Cancelled,
        InvalidInput,
        UnsupportedConfiguration,
        BackendUnavailable,
        NumericalFailure,
    };

    struct BackendCapabilities
    {
        bool optimizesPoints = false;
        bool refinesCameraPose = false;
        bool refinesSharedFocalLength = false;
        bool refinesSharedFocalAspectRatio = false;
        bool refinesSharedPrincipalPoint = false;
        bool refinesSharedRadialDistortion = false;
        bool supportsSoftConstraints = false;
        bool supportsLaserRangeConstraints = false;
    };

    struct BackendDecision
    {
        Backend backend = Backend::PlaMatrixCpu;
        std::string reason;
    };

    const char* backendName(Backend backend) noexcept;
    const char* solveStatusName(SolveStatus status) noexcept;

} // namespace plabundle
