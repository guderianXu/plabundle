#pragma once

#include <plabundle/camera.h>
#include <plabundle/constraints.h>
#include <plabundle/rig.h>

#include <optional>
#include <string>
#include <vector>

namespace plabundle
{

    enum class GaugePolicy
    {
        AutoAnchor,
        RequireExplicitGauge,
        CallerManaged,
    };

    struct Gauge
    {
        GaugePolicy policy = GaugePolicy::AutoAnchor;
        int referenceAnchorCameraIndex = -1;
        int referenceScaleCameraIndex = -1;
        double referenceBaseline = 0.0;
    };

    struct ProblemStats
    {
        int cameraCount = 0;
        int trackCount = 0;
        int observationCount = 0;
    };

    struct Problem
    {
        std::vector<FrameCamera> cameras;
        std::vector<Track> tracks;
        std::vector<int> fixedCameraIndices;
        std::vector<int> fixedTrackIndices;
        // Cameras with the same id share one intrinsic/model parameter block.
        // Unique ids give each camera an independent block; disabling all
        // intrinsic parameters keeps calibration fixed.
        std::vector<int> cameraCalibrationGroupIds;
        std::vector<FrameCamera> sharedIntrinsicReferenceCameras;
        std::vector<LaserRangeConstraint> laserRangeConstraints;
        std::vector<ScaleBarConstraint> scaleBarConstraints;
        std::vector<std::optional<CameraPosePrior>> cameraPosePriors;
        std::optional<CameraPlaneConstraint> cameraPlaneConstraint;
        RigTopology rig;
        Gauge gauge;
    };

    ProblemStats summarizeProblem(const Problem& problem);
    bool validateProblem(const Problem& problem, std::string* error = nullptr);

} // namespace plabundle
