#pragma once

#include <array>
#include <string>
#include <vector>

namespace plabundle
{

    enum class PosePriorComponents
    {
        Position,
        Rotation,
        RotationAndPosition,
    };

    enum class PosePriorUncertainty
    {
        IndependentSigmas,
        Covariance,
        SqrtInformation,
    };

    enum class PosePriorTangentFrame
    {
        World,
        PriorCamera,
    };

    struct Observation
    {
        int cameraIndex = -1;
        double u = 0.0;
        double v = 0.0;
        double weight = 1.0;
        double measurementScale = 1.0;
    };

    struct LaserPlaneConstraint
    {
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        std::array<double, 3> normal{{0.0, 0.0, 1.0}};
        double weight = 1.0;
        double initialSignedDistance = 0.0;
        int sourceFrameIndex = -1;
    };

    enum class LaserPointMode
    {
        Unspecified,
        Fixed,
        Constrained,
        Free,
    };

    struct LaserRangeConstraint
    {
        int cameraIndex = -1;
        std::array<double, 3> initialPoint{{0.0, 0.0, 0.0}};
        double observedRangeMeters = 0.0;
        double sigmaRangeMeters = 1.0;
        double weight = 1.0;
        std::array<double, 3> leverArmCameraMeters{{0.0, 0.0, 0.0}};
        LaserPointMode pointMode = LaserPointMode::Unspecified;
        std::array<double, 3> pointPrior{{0.0, 0.0, 0.0}};
        std::array<double, 9> pointPriorSqrtInformation{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        std::vector<Observation> measuredImageObservations;
        std::string shotId;
        double ephemerisTimeSeconds = 0.0;
        int sourceIndex = -1;
    };

    struct ControlPointConstraint
    {
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        double sigmaMeters = 1.0;
        double weight = 1.0;
        int sourceIndex = -1;
        enum class Uncertainty
        {
            IsotropicSigma,
            Covariance,
            SqrtInformation,
        } uncertainty = Uncertainty::IsotropicSigma;
        // Row-major 3x3 covariance (m^2) or square-root information (m^-1) in the solver frame.
        std::array<double, 9> uncertaintyMatrix{};
    };

    using ControlPointUncertainty = ControlPointConstraint::Uncertainty;

    bool validateControlPointConstraint(const ControlPointConstraint& constraint, std::string* error = nullptr);

    struct ScaleBarConstraint
    {
        int trackIndexA = -1;
        int trackIndexB = -1;
        double measuredDistanceMeters = 0.0;
        double sigmaMeters = 1.0;
        double weight = 1.0;
        int sourceIndex = -1;
    };

    struct CameraPosePrior
    {
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
        double positionSigmaMeters = 1.0;
        double rotationSigmaDegrees = 2.0;
        PosePriorComponents components = PosePriorComponents::RotationAndPosition;
        PosePriorUncertainty uncertainty = PosePriorUncertainty::IndependentSigmas;
        PosePriorTangentFrame tangentFrame = PosePriorTangentFrame::World;
        // Row-major 6x6 in [rotation radians, position meters] order.
        std::array<double, 36> uncertaintyMatrix{};
    };

    bool validateCameraPosePrior(const CameraPosePrior& prior, std::string* error = nullptr);

    struct CameraPlaneConstraint
    {
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        std::array<double, 3> normal{{0.0, 0.0, 1.0}};
        std::vector<double> referenceSignedDistances;
        double sigmaMeters = 1.0;
        double weight = 1.0;
    };

    struct Track
    {
        std::array<double, 3> initialPoint{{0.0, 0.0, 0.0}};
        std::vector<Observation> observations;
        std::vector<LaserPlaneConstraint> laserPlaneConstraints;
        std::vector<ControlPointConstraint> controlPointConstraints;
    };

} // namespace plabundle
