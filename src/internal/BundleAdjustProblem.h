#pragma once

#include "BundleAdjustTypes.h"

#include <plabundle/constraints.h>

#include <array>
#include <vector>

namespace plabundle::internal
{

    using BAObservation = Observation;
    using BALaserPlaneConstraint = LaserPlaneConstraint;
    using BALaserPointMode = LaserPointMode;
    using BALaserRangeConstraint = LaserRangeConstraint;
    using BAControlPointConstraint = ControlPointConstraint;
    using BAScaleBarConstraint = ScaleBarConstraint;
    using BATrack = Track;

    struct BACameraPosePrior
    {
        bool enabled = false;
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
        double positionSigmaMeters = 1.0;
        double rotationSigmaDegrees = 2.0;
    };

    struct BACameraPlaneConstraint
    {
        bool enabled = false;
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        std::array<double, 3> normal{{0.0, 0.0, 1.0}};
        std::vector<double> referenceSignedDistances;
        double sigmaMeters = 1.0;
        double weight = 1.0;
    };

} // namespace plabundle::internal
