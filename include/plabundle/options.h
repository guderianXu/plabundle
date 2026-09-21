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
        Count,
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

    struct Options
    {
        static constexpr int kAutoBackendPolicyVersion = 2;
        static constexpr int kDefaultMinPlaMatrixCudaCameras = 128;
        static constexpr int kDefaultMinPlaMatrixCudaObservations = 30000;
        static constexpr int kDefaultMinPlaMatrixOpenClCameras = 160;
        static constexpr int kDefaultMinPlaMatrixOpenClObservations = 50000;
        static constexpr int kDefaultMinPlaMatrixDenseCameras = 120;
        static constexpr int kDefaultMinPlaMatrixCudaDenseObservations = 150000;
        static constexpr int kDefaultMinPlaMatrixOpenClDenseObservations = 200000;

        Backend backend = Backend::PlaMatrixCpu;
        int maxIterations = 20;
        bool refineCameraPose = true;
        bool refineSharedFocalLength = false;
        bool refineSharedFocalAspectRatio = false;
        bool refineSharedPrincipalPoint = false;
        bool refineSharedRadialDistortion = false;
        bool refineSharedHighOrderDistortion = true;
        bool hasTrustedSharedFocalPrior = false;
        bool useSharedIntrinsicParameterMask = false;
        IntrinsicParameterMask sharedIntrinsicParameterMask{{true, true, true, true, true, true, true, true, true}};
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
        double sharedRadialK1PriorSigma = 0.15;
        double sharedRadialK2PriorSigma = 0.15;
        double sharedRadialK3PriorSigma = 0.15;
        double sharedTangentialP1PriorSigma = 0.01;
        double sharedTangentialP2PriorSigma = 0.01;
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
        int minPlaMatrixOpenClCameras = kDefaultMinPlaMatrixOpenClCameras;
        int minPlaMatrixOpenClObservations = kDefaultMinPlaMatrixOpenClObservations;
        int minPlaMatrixDenseCameras = kDefaultMinPlaMatrixDenseCameras;
        int minPlaMatrixCudaDenseObservations = kDefaultMinPlaMatrixCudaDenseObservations;
        int minPlaMatrixOpenClDenseObservations = kDefaultMinPlaMatrixOpenClDenseObservations;
        double maxInitialTrackRms = 100.0;
        int plaMatrixPreconditionerClusterSize = 1;
        // Explicit unavailable or unusable accelerated backends may rerun on PlaMatrixCpu when enabled.
        bool allowBackendFallback = true;
        // Auto candidates are checked before publication; rejected accelerated candidates rerun on PlaMatrixCpu.
        bool enableBackendQualityGate = true;
        double maxAcceptedRmsGrowth = 1.25;
        double minAcceptedValidTrackRatio = 0.60;
        double maxAcceptedConstraintRmsGrowth = 1.25;
        std::stop_token stopToken;
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        std::function<bool(const IterationSummary&)> progressCallback;
    };

    bool sharedIntrinsicParameterEnabled(const Options& options, IntrinsicParameter parameter) noexcept;
    bool validateOptions(const Options& options, std::string* error = nullptr) noexcept;

} // namespace plabundle
