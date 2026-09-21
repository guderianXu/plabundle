#pragma once

#include "BundleAdjustOptions.h"
#include "BundleAdjustProblem.h"
#include "BundleAdjustTypes.h"
#include "CameraState.h"

#include <plabundle/adaptive_camera_model.h>

#include <string>
#include <vector>

namespace plabundle::internal
{

    using BAAdaptiveCameraModelAssessment = AdaptiveCameraModelAssessment;

    const char* baIntrinsicParameterName(BAIntrinsicParameter parameter);
    int enabledIntrinsicParameterCount(const BAIntrinsicParameterMask& mask);
    std::string adaptiveCameraModelName(const BAIntrinsicParameterMask& mask);
    BAAdaptiveCameraModelAssessment assessAdaptiveCameraModel(const std::vector<CameraState>& cameras,
                                                              const std::vector<BATrack>& tracks,
                                                              const BAOptions* options = nullptr);
    bool applyAdaptiveCameraModel(const BAAdaptiveCameraModelAssessment& assessment, BAOptions* options);
    bool restoreInactiveAdaptiveIntrinsics(std::vector<CameraState>* cameras,
                                           const std::vector<CameraState>& stableReferences,
                                           const BAIntrinsicParameterMask& activeMask);

} // namespace plabundle::internal
