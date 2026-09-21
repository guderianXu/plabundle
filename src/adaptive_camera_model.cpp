#include <plabundle/adaptive_camera_model.h>

#include "internal/BundleAdjustAdaptiveCameraModel.h"
#include "internal/Conversion.h"

namespace plabundle
{

    const char* intrinsicParameterName(IntrinsicParameter parameter) noexcept
    {
        return internal::baIntrinsicParameterName(parameter);
    }

    int enabledIntrinsicParameterCount(const IntrinsicParameterMask& mask) noexcept
    {
        return internal::enabledIntrinsicParameterCount(mask);
    }

    std::string adaptiveCameraModelName(const IntrinsicParameterMask& mask)
    {
        return internal::adaptiveCameraModelName(mask);
    }

    AdaptiveCameraModelAssessment assessAdaptiveCameraModel(const Problem& problem, const Options* options)
    {
        const Options default_options;
        const Options& requested_options = options ? *options : default_options;
        const internal::BAOptions solver_options =
            internal::makeSolverOptions(problem, requested_options, Backend::PlaMatrixCpu);
        return internal::assessAdaptiveCameraModel(
            internal::makeCameraStates(problem.cameras), problem.tracks, &solver_options);
    }

    bool applyAdaptiveCameraModel(const AdaptiveCameraModelAssessment& assessment, Options* options)
    {
        if (!options)
        {
            return false;
        }
        internal::BAOptions solver_options;
        static_cast<Options&>(solver_options) = *options;
        const bool applied = internal::applyAdaptiveCameraModel(assessment, &solver_options);
        *options = static_cast<const Options&>(solver_options);
        return applied;
    }

    bool restoreInactiveAdaptiveIntrinsics(std::vector<FrameCamera>* cameras,
                                           const std::vector<FrameCamera>& stableReferences,
                                           const IntrinsicParameterMask& activeMask)
    {
        if (!cameras)
        {
            return false;
        }
        std::vector<internal::CameraState> camera_states = internal::makeCameraStates(*cameras);
        const std::vector<internal::CameraState> references = internal::makeCameraStates(stableReferences);
        if (!internal::restoreInactiveAdaptiveIntrinsics(&camera_states, references, activeMask))
        {
            return false;
        }
        *cameras = internal::makeFrameCameras(camera_states);
        return true;
    }

} // namespace plabundle
