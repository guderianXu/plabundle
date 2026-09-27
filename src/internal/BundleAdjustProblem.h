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

    struct BACameraPosePrior : CameraPosePrior
    {
        bool enabled = false;
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
