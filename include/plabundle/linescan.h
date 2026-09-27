#pragma once

#include <plabundle/backend.h>
#include <plabundle/options.h>

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace plabundle::linescan
{

    struct ImageObservation
    {
        int cameraIndex = -1;
        int pointIndex = -1;
        double samplePixels = 0.0;
        double linePixels = 0.0;
    };

    struct SensorPose
    {
        std::array<double, 3> centerMeters{};
        std::array<double, 9> worldToSensorRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    };

    // Implementations own reusable trajectory/spline/ephemeris state. evaluate()
    // must be safe for concurrent const calls when Options::threadCount > 1.
    class PoseTrajectory
    {
    public:
        virtual ~PoseTrajectory() = default;
        virtual bool evaluate(double timeSeconds, SensorPose* pose) const = 0;
    };

    struct LineTimingModel
    {
        double referenceLinePixels = 0.0;
        double referenceTimeSeconds = 0.0;
        double secondsPerLine = 1.0;

        double timeAtLine(double linePixels) const noexcept;
    };

    struct CameraModel
    {
        LineTimingModel timing;
        std::shared_ptr<const PoseTrajectory> trajectory;
        double focalSamplePixels = 1.0;
        double focalLinePixels = 1.0;
        double principalSamplePixels = 0.0;
        double detectorLinePixels = 0.0;
        double minimumDepthMeters = 1.0e-8;
    };

    // This state is independent of the optimized correction and point. A caller
    // may prepare it once per observation and reuse it for residual/Jacobian
    // evaluations at the same image line.
    struct ProjectionState
    {
        double acquisitionTimeSeconds = 0.0;
        SensorPose nominalPose;
    };

    struct ProjectionEvaluation
    {
        std::array<double, 2> residualPixels{};
        std::array<double, 12> cameraJacobian{};
        std::array<double, 6> pointJacobian{};
    };

    bool prepareProjectionState(const CameraModel& model, double linePixels, ProjectionState* state);
    bool evaluateProjection(const CameraModel& model,
                            const ProjectionState& state,
                            const ImageObservation& observation,
                            const std::array<double, 6>& cameraCorrection,
                            const std::array<double, 3>& pointMeters,
                            ProjectionEvaluation* evaluation);

    enum class LaserPointMode
    {
        Fixed,
        Constrained,
    };

    struct LaserPoint
    {
        std::array<double, 3> initialMeters{};
        std::array<double, 3> refinedMeters{};
        std::array<double, 9> sqrtInformation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        LaserPointMode mode = LaserPointMode::Fixed;
    };

    struct LaserObservation
    {
        int cameraIndex = -1;
        int laserPointIndex = -1;
        std::array<double, 3> nominalSensorCenterMeters{};
        double observedRangeMeters = 0.0;
        double sigmaMeters = 1.0;
    };

    // The caller owns camera/trajectory models. A callback evaluates the two raw
    // pixel residuals at the observed line; it must remain valid for solve()
    // and support concurrent calls when Options::threadCount > 1.
    using Projection = std::function<bool(
        const ImageObservation&, const std::array<double, 6>&, const std::array<double, 3>&, std::array<double, 2>*)>;

    struct Problem
    {
        std::vector<std::array<double, 6>> cameraParameters;
        std::vector<CameraModel> cameraModels;
        std::vector<std::array<double, 3>> tiePoints;
        std::vector<ImageObservation> imageObservations;
        std::vector<LaserPoint> laserPoints;
        std::vector<LaserObservation> laserObservations;
        Projection projection;
    };

    enum class DiagnosticCode
    {
        UnobservedCamera,
        SingleViewPoint,
        InvalidCameraModel,
        DegenerateProjection,
    };

    struct Diagnostic
    {
        DiagnosticCode code = DiagnosticCode::DegenerateProjection;
        int cameraIndex = -1;
        int pointIndex = -1;
        std::string message;
    };

    struct Options
    {
        Backend backend = Backend::Auto;
        int plaMatrixDevice = 0;
        int minPlaMatrixCudaCameras = plabundle::Options::kDefaultMinPlaMatrixCudaCameras;
        int minPlaMatrixCudaObservations = plabundle::Options::kDefaultMinPlaMatrixCudaObservations;
        int minPlaMatrixVulkanCameras = plabundle::Options::kDefaultMinPlaMatrixVulkanCameras;
        int minPlaMatrixVulkanObservations = plabundle::Options::kDefaultMinPlaMatrixVulkanObservations;
        int minPlaMatrixOpenClCameras = plabundle::Options::kDefaultMinPlaMatrixOpenClCameras;
        int minPlaMatrixOpenClObservations = plabundle::Options::kDefaultMinPlaMatrixOpenClObservations;
        int minPlaMatrixDenseCameras = plabundle::Options::kDefaultMinPlaMatrixDenseCameras;
        int minPlaMatrixCudaDenseObservations = plabundle::Options::kDefaultMinPlaMatrixCudaDenseObservations;
        int minPlaMatrixVulkanDenseObservations = plabundle::Options::kDefaultMinPlaMatrixVulkanDenseObservations;
        int minPlaMatrixOpenClDenseObservations = plabundle::Options::kDefaultMinPlaMatrixOpenClDenseObservations;
        int maxDenseSchurCameras = 200;
        bool allowBackendFallback = true;
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        bool enableLaserRangeConstraints = false;
        int maximumIterations = 50;
        int threadCount = 0;
        double imageSigmaPixels = 1.0;
        double imageHuberDeltaPixels = 3.0;
        double cameraPositionSigmaMeters = 1000.0;
        double cameraAngleSigmaDegrees = 2.0;
        double laserRangeWeight = 1.0;
        double laserRangeHuberDeltaSigma = 3.0;
        double finiteDifferencePointStepMeters = 0.05;
        double finiteDifferencePositionStepMeters = 0.05;
        double finiteDifferenceAngleStepRadians = 1.0e-7;
        double maximumCameraTranslationMeters = 5000.0;
        double maximumCameraAngleDegrees = 5.0;
    };

    struct Result
    {
        bool success = false;
        bool solutionUsable = false;
        bool converged = false;
        bool backendFallback = false;
        bool usedGpu = false;
        Backend requestedBackend = Backend::Auto;
        Backend usedBackend = Backend::Auto;
        std::string terminationType;
        std::string message;
        std::string backendMessage;
        std::string linearSolverName;
        std::string deviceName;
        std::string solverBriefReport;
        int iterations = 0;
        double initialImageRmsPixels = 0.0;
        double refinedImageRmsPixels = 0.0;
        double initialLaserRangeRmsMeters = 0.0;
        double refinedLaserRangeRmsMeters = 0.0;
        std::vector<std::array<double, 6>> refinedCameraParameters;
        std::vector<std::array<double, 3>> refinedTiePoints;
        std::vector<std::array<double, 3>> refinedLaserPoints;
        std::vector<Diagnostic> diagnostics;

        bool usable() const noexcept
        {
            return success && solutionUsable;
        }
    };

    Result solve(const Problem& problem, const Options& options = Options());

    // Compatibility entry point. New callers should use the immutable overload above
    // and consume the refined state carried by Result.
    bool solve(Problem* problem, const Options& options, Result* result, std::string* errorMessage = nullptr);

} // namespace plabundle::linescan
