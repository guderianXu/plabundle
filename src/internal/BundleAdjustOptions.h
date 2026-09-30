#pragma once

#include "BundleAdjustProblem.h"
#include "CameraState.h"

#include <plabundle/options.h>
#include <placamera/camera_topology.h>

#include <vector>

namespace plabundle::internal
{

    struct BAOptions : Options
    {
        std::vector<int> cameraCalibrationGroupIds;
        std::vector<CameraState> sharedIntrinsicReferenceCameras;
        int referenceGaugeAnchorCameraIndex = -1;
        int referenceGaugeScaleCameraIndex = -1;
        double referenceGaugeBaseline = 0.0;
        bool enableLaserPlaneConstraints = false;
        bool enableLaserRangeConstraints = false;
        std::vector<BALaserRangeConstraint> laserRangeConstraints;
        bool enableControlPointConstraints = false;
        bool enableScaleBarConstraints = false;
        std::vector<BAScaleBarConstraint> scaleBarConstraints;
        std::vector<BACameraPosePrior> cameraPosePriors;
        BACameraPlaneConstraint cameraPlaneConstraint;
        BAGaugePolicy gaugePolicy = BAGaugePolicy::AutoAnchor;
        std::vector<int> fixedCameraIndices;
        std::vector<int> fixedTrackIndices;
        placamera::RigTopology rig;
    };

    inline bool sharedIntrinsicParameterEnabled(const BAOptions& options, BAIntrinsicParameter parameter) noexcept
    {
        return plabundle::sharedIntrinsicParameterEnabled(options, parameter);
    }

} // namespace plabundle::internal
