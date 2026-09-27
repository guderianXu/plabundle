#pragma once

#include <plabundle/backend.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stop_token>
#include <string>

namespace plabundle
{

    enum class ImageRobustLoss : std::uint8_t
    {
        LeastSquares,
        Huber,
        Cauchy,
    };

    enum class IntrinsicParameter : std::uint8_t
    {
        FocalLength = 0,
        FocalAspectRatio,
        PrincipalPointX,
        PrincipalPointY,
        RadialK1,
        RadialK2,
        RadialK3,
        TangentialP1,
        TangentialP2,
        SkewB2,
        RadialK4,
        TangentialP3,
        TangentialP4,
        ModelCoefficient0 = RadialK1,
        ModelCoefficient1 = RadialK2,
        ModelCoefficient2 = RadialK3,
        ModelCoefficient3 = TangentialP1,
        ModelCoefficient4 = TangentialP2,
        Count = 13,
    };

    inline constexpr std::size_t kIntrinsicParameterCount = static_cast<std::size_t>(IntrinsicParameter::Count);
    using IntrinsicParameterMask = std::array<bool, kIntrinsicParameterCount>;

    struct IterationSummary
    {
        int currentIteration = 0;
        int maxIterations = 0;
        double averageRmsPixels = 0.0;
        int validPointCount = 0;
    };

    struct SolverControlOptions
    {
        int maxIterations = 20;
        ImageRobustLoss imageRobustLoss = ImageRobustLoss::LeastSquares;
        double imageRobustLossScalePixels = 1.0;
        double referenceArmijoCoefficient = 1.0e-3;
        int referenceLineSearchSteps = 34;
        bool useReferenceOnlineSchur = false;
        bool enablePointFilter = true;
        double filterMaxReprojError = 2.5;
        double filterSigmaFactor = 3.0;
        double maxInitialTrackRms = 100.0;
        int numThreads = 0;
        bool logIterationProgress = true;
        std::stop_token stopToken;
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        std::function<bool(const IterationSummary&)> progressCallback;
    };

    struct CalibrationOptions
    {
        bool refineCameraPose = true;
        bool refineSharedFocalLength = false;
        bool refineSharedFocalAspectRatio = false;
        bool refineSharedPrincipalPoint = false;
        bool refineSharedModelCoefficients = false;
        bool refineSharedRadialDistortion = false;
        bool refineSharedHighOrderDistortion = true;
        bool refineSharedMetashapeParameters = false;
        bool hasTrustedSharedFocalPrior = false;
        bool useSharedIntrinsicParameterMask = false;
        IntrinsicParameterMask sharedIntrinsicParameterMask{
            {true, true, true, true, true, true, true, true, true, true, true, true, true}};
        double minSharedFocalScale = 0.5;
        double maxSharedFocalScale = 4.0;
        double minSharedFocalAspectScale = 0.85;
        double maxSharedFocalAspectScale = 1.18;
        double maxSharedPrincipalPointOffsetFraction = 0.08;
        double sharedFocalPriorSigma = 0.35;
        double sharedPrincipalPointPriorSigmaFraction = 0.04;
        double sharedFocalAspectPriorSigma = 0.08;
        double maxSharedRadialK1Abs = 0.35;
        double maxSharedRadialK2Abs = 0.35;
        double maxSharedRadialK3Abs = 0.35;
        double maxSharedTangentialP1Abs = 0.02;
        double maxSharedTangentialP2Abs = 0.02;
        double maxSharedSkewFraction = 0.10;
        double maxSharedRadialK4Abs = 0.35;
        double maxSharedTangentialP3Abs = 0.35;
        double maxSharedTangentialP4Abs = 0.35;
        double sharedRadialK1PriorSigma = 0.15;
        double sharedRadialK2PriorSigma = 0.15;
        double sharedRadialK3PriorSigma = 0.15;
        double sharedTangentialP1PriorSigma = 0.01;
        double sharedTangentialP2PriorSigma = 0.01;
        double sharedSkewPriorSigmaFraction = 0.02;
        double sharedRadialK4PriorSigma = 0.15;
        double sharedTangentialP3PriorSigma = 0.15;
        double sharedTangentialP4PriorSigma = 0.15;
        double sharedLowOrderDistortionScale = 1.0;
        bool useReferenceCalibrationTransitionPrior = false;
        IntrinsicParameterMask referencePreviousIntrinsicParameterMask{};
    };

    struct ConstraintOptions
    {
        double laserPlaneWeight = 1.0;
        double laserHuberDeltaMeters = 0.2;
        double laserRangeWeight = 1.0;
        double laserRangeHuberDelta = 3.0;
        double controlPointWeight = 1.0;
        double controlPointHuberDeltaMeters = 0.2;
        double scaleBarWeight = 1.0;
        double scaleBarHuberDeltaMeters = 0.2;
        double cameraPosePriorWeight = 1000.0;
        double cameraPosePriorHuberDelta = 3.0;
        double cameraPlaneHuberDelta = 3.0;
    };

    struct BackendOptions
    {
        static constexpr int kAutoPolicyVersion = 3;
        static constexpr int kDefaultMinCudaCameras = 128;
        static constexpr int kDefaultMinCudaObservations = 30000;
        static constexpr int kDefaultMinVulkanCameras = 160;
        static constexpr int kDefaultMinVulkanObservations = 50000;
        static constexpr int kDefaultMinOpenClCameras = 160;
        static constexpr int kDefaultMinOpenClObservations = 50000;
        static constexpr int kDefaultMinDenseCameras = 120;
        static constexpr int kDefaultMinCudaDenseObservations = 150000;
        static constexpr int kDefaultMinVulkanDenseObservations = 200000;
        static constexpr int kDefaultMinOpenClDenseObservations = 200000;

        Backend requested = Backend::PlaMatrixCpu;
        int plaMatrixDevice = 0;
        bool enablePlaMatrixMixedPrecision = false;
        int minPlaMatrixCudaCameras = kDefaultMinCudaCameras;
        int minPlaMatrixCudaObservations = kDefaultMinCudaObservations;
        int minPlaMatrixVulkanCameras = kDefaultMinVulkanCameras;
        int minPlaMatrixVulkanObservations = kDefaultMinVulkanObservations;
        int minPlaMatrixOpenClCameras = kDefaultMinOpenClCameras;
        int minPlaMatrixOpenClObservations = kDefaultMinOpenClObservations;
        int minPlaMatrixDenseCameras = kDefaultMinDenseCameras;
        int minPlaMatrixCudaDenseObservations = kDefaultMinCudaDenseObservations;
        int minPlaMatrixVulkanDenseObservations = kDefaultMinVulkanDenseObservations;
        int minPlaMatrixOpenClDenseObservations = kDefaultMinOpenClDenseObservations;
        int plaMatrixPreconditionerClusterSize = 1;
        bool allowFallback = true;
    };

    struct QualityGateOptions
    {
        bool enabled = true;
        double maxAcceptedRmsGrowth = 1.25;
        double minAcceptedValidTrackRatio = 0.60;
        double maxAcceptedConstraintRmsGrowth = 1.25;
    };

    struct SolveOptions
    {
        SolverControlOptions solver;
        CalibrationOptions calibration;
        ConstraintOptions constraints;
        BackendOptions backend;
        QualityGateOptions quality;
    };

    // Compatibility layout retained for existing callers. New code should use SolveOptions.
    struct Options
    {
        static constexpr int kAutoBackendPolicyVersion = BackendOptions::kAutoPolicyVersion;
        static constexpr int kDefaultMinPlaMatrixCudaCameras = BackendOptions::kDefaultMinCudaCameras;
        static constexpr int kDefaultMinPlaMatrixCudaObservations = BackendOptions::kDefaultMinCudaObservations;
        static constexpr int kDefaultMinPlaMatrixVulkanCameras = BackendOptions::kDefaultMinVulkanCameras;
        static constexpr int kDefaultMinPlaMatrixVulkanObservations = BackendOptions::kDefaultMinVulkanObservations;
        static constexpr int kDefaultMinPlaMatrixOpenClCameras = BackendOptions::kDefaultMinOpenClCameras;
        static constexpr int kDefaultMinPlaMatrixOpenClObservations = BackendOptions::kDefaultMinOpenClObservations;
        static constexpr int kDefaultMinPlaMatrixDenseCameras = BackendOptions::kDefaultMinDenseCameras;
        static constexpr int kDefaultMinPlaMatrixCudaDenseObservations =
            BackendOptions::kDefaultMinCudaDenseObservations;
        static constexpr int kDefaultMinPlaMatrixVulkanDenseObservations =
            BackendOptions::kDefaultMinVulkanDenseObservations;
        static constexpr int kDefaultMinPlaMatrixOpenClDenseObservations =
            BackendOptions::kDefaultMinOpenClDenseObservations;

        Backend backend = Backend::PlaMatrixCpu;
        int maxIterations = 20;
        ImageRobustLoss imageRobustLoss = ImageRobustLoss::LeastSquares;
        double imageRobustLossScalePixels = 1.0;
        bool refineCameraPose = true;
        bool refineSharedFocalLength = false;
        bool refineSharedFocalAspectRatio = false;
        bool refineSharedPrincipalPoint = false;
        bool refineSharedModelCoefficients = false;
        bool refineSharedRadialDistortion = false;
        bool refineSharedHighOrderDistortion = true;
        bool refineSharedMetashapeParameters = false;
        bool hasTrustedSharedFocalPrior = false;
        bool useSharedIntrinsicParameterMask = false;
        IntrinsicParameterMask sharedIntrinsicParameterMask{
            {true, true, true, true, true, true, true, true, true, true, true, true, true}};
        double minSharedFocalScale = 0.5;
        double maxSharedFocalScale = 4.0;
        double minSharedFocalAspectScale = 0.85;
        double maxSharedFocalAspectScale = 1.18;
        double maxSharedPrincipalPointOffsetFraction = 0.08;
        double sharedFocalPriorSigma = 0.35;
        double sharedPrincipalPointPriorSigmaFraction = 0.04;
        double sharedFocalAspectPriorSigma = 0.08;
        double maxSharedRadialK1Abs = 0.35;
        double maxSharedRadialK2Abs = 0.35;
        double maxSharedRadialK3Abs = 0.35;
        double maxSharedTangentialP1Abs = 0.02;
        double maxSharedTangentialP2Abs = 0.02;
        double maxSharedSkewFraction = 0.10;
        double maxSharedRadialK4Abs = 0.35;
        double maxSharedTangentialP3Abs = 0.35;
        double maxSharedTangentialP4Abs = 0.35;
        double sharedRadialK1PriorSigma = 0.15;
        double sharedRadialK2PriorSigma = 0.15;
        double sharedRadialK3PriorSigma = 0.15;
        double sharedTangentialP1PriorSigma = 0.01;
        double sharedTangentialP2PriorSigma = 0.01;
        double sharedSkewPriorSigmaFraction = 0.02;
        double sharedRadialK4PriorSigma = 0.15;
        double sharedTangentialP3PriorSigma = 0.15;
        double sharedTangentialP4PriorSigma = 0.15;
        double sharedLowOrderDistortionScale = 1.0;
        double referenceArmijoCoefficient = 1.0e-3;
        int referenceLineSearchSteps = 34;
        bool useReferenceOnlineSchur = false;
        bool useReferenceCalibrationTransitionPrior = false;
        IntrinsicParameterMask referencePreviousIntrinsicParameterMask{};
        double laserPlaneWeight = 1.0;
        double laserHuberDeltaMeters = 0.2;
        double laserRangeWeight = 1.0;
        double laserRangeHuberDelta = 3.0;
        double controlPointWeight = 1.0;
        double controlPointHuberDeltaMeters = 0.2;
        double scaleBarWeight = 1.0;
        double scaleBarHuberDeltaMeters = 0.2;
        double cameraPosePriorWeight = 1000.0;
        double cameraPosePriorHuberDelta = 3.0;
        double cameraPlaneHuberDelta = 3.0;
        bool enablePointFilter = true;
        double filterMaxReprojError = 2.5;
        double filterSigmaFactor = 3.0;
        int numThreads = 0;
        bool logIterationProgress = true;
        // CUDA uses this index directly. OpenCL requires it to match the process-wide device selected by PlaMatrix.
        int plaMatrixDevice = 0;
        // Requests a guarded FP32 PCG seed; diagnostics report true only when the seed is accepted.
        bool enablePlaMatrixMixedPrecision = false;
        int minPlaMatrixCudaCameras = kDefaultMinPlaMatrixCudaCameras;
        int minPlaMatrixCudaObservations = kDefaultMinPlaMatrixCudaObservations;
        int minPlaMatrixVulkanCameras = kDefaultMinPlaMatrixVulkanCameras;
        int minPlaMatrixVulkanObservations = kDefaultMinPlaMatrixVulkanObservations;
        int minPlaMatrixOpenClCameras = kDefaultMinPlaMatrixOpenClCameras;
        int minPlaMatrixOpenClObservations = kDefaultMinPlaMatrixOpenClObservations;
        int minPlaMatrixDenseCameras = kDefaultMinPlaMatrixDenseCameras;
        int minPlaMatrixCudaDenseObservations = kDefaultMinPlaMatrixCudaDenseObservations;
        int minPlaMatrixVulkanDenseObservations = kDefaultMinPlaMatrixVulkanDenseObservations;
        int minPlaMatrixOpenClDenseObservations = kDefaultMinPlaMatrixOpenClDenseObservations;
        double maxInitialTrackRms = 100.0;
        int plaMatrixPreconditionerClusterSize = 1;
        // Explicit unavailable, unusable, or quality-rejected accelerated backends may rerun on PlaMatrixCpu.
        bool allowBackendFallback = true;
        // Every concrete candidate, including CPU fallbacks, is checked before publication.
        bool enableBackendQualityGate = true;
        double maxAcceptedRmsGrowth = 1.25;
        double minAcceptedValidTrackRatio = 0.60;
        double maxAcceptedConstraintRmsGrowth = 1.25;
        std::stop_token stopToken;
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        std::function<bool(const IterationSummary&)> progressCallback;
    };

    Options makeCompatibilityOptions(const SolveOptions& options);
    SolveOptions makeSolveOptions(const Options& options);
    bool sharedIntrinsicParameterEnabled(const Options& options, IntrinsicParameter parameter) noexcept;
    bool sharedIntrinsicParameterEnabled(const SolveOptions& options, IntrinsicParameter parameter) noexcept;
    bool validateOptions(const Options& options, std::string* error = nullptr) noexcept;
    bool validateOptions(const SolveOptions& options, std::string* error = nullptr) noexcept;

} // namespace plabundle
