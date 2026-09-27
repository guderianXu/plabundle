#include <plabundle/options.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace plabundle
{
    namespace
    {
        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        void setError(std::string* error, const char* message) noexcept
        {
            if (!error)
            {
                return;
            }
            try
            {
                *error = message;
            }
            catch (...)
            {
            }
        }

        bool positive(double value) noexcept
        {
            return finite(value) && value > 0.0;
        }

        bool nonNegative(double value) noexcept
        {
            return finite(value) && value >= 0.0;
        }

    } // namespace

    Options makeCompatibilityOptions(const SolveOptions& options)
    {
        Options compatibility;

        compatibility.maxIterations = options.solver.maxIterations;
        compatibility.imageRobustLoss = options.solver.imageRobustLoss;
        compatibility.imageRobustLossScalePixels = options.solver.imageRobustLossScalePixels;
        compatibility.referenceArmijoCoefficient = options.solver.referenceArmijoCoefficient;
        compatibility.referenceLineSearchSteps = options.solver.referenceLineSearchSteps;
        compatibility.useReferenceOnlineSchur = options.solver.useReferenceOnlineSchur;
        compatibility.enablePointFilter = options.solver.enablePointFilter;
        compatibility.filterMaxReprojError = options.solver.filterMaxReprojError;
        compatibility.filterSigmaFactor = options.solver.filterSigmaFactor;
        compatibility.maxInitialTrackRms = options.solver.maxInitialTrackRms;
        compatibility.numThreads = options.solver.numThreads;
        compatibility.logIterationProgress = options.solver.logIterationProgress;
        compatibility.stopToken = options.solver.stopToken;
        compatibility.cancelFlag = options.solver.cancelFlag;
        compatibility.progressCallback = options.solver.progressCallback;

        compatibility.refineCameraPose = options.calibration.refineCameraPose;
        compatibility.refineSharedFocalLength = options.calibration.refineSharedFocalLength;
        compatibility.refineSharedFocalAspectRatio = options.calibration.refineSharedFocalAspectRatio;
        compatibility.refineSharedPrincipalPoint = options.calibration.refineSharedPrincipalPoint;
        compatibility.refineSharedModelCoefficients = options.calibration.refineSharedModelCoefficients;
        compatibility.refineSharedRadialDistortion = options.calibration.refineSharedRadialDistortion;
        compatibility.refineSharedHighOrderDistortion = options.calibration.refineSharedHighOrderDistortion;
        compatibility.refineSharedMetashapeParameters = options.calibration.refineSharedMetashapeParameters;
        compatibility.hasTrustedSharedFocalPrior = options.calibration.hasTrustedSharedFocalPrior;
        compatibility.useSharedIntrinsicParameterMask = options.calibration.useSharedIntrinsicParameterMask;
        compatibility.sharedIntrinsicParameterMask = options.calibration.sharedIntrinsicParameterMask;
        compatibility.minSharedFocalScale = options.calibration.minSharedFocalScale;
        compatibility.maxSharedFocalScale = options.calibration.maxSharedFocalScale;
        compatibility.minSharedFocalAspectScale = options.calibration.minSharedFocalAspectScale;
        compatibility.maxSharedFocalAspectScale = options.calibration.maxSharedFocalAspectScale;
        compatibility.maxSharedPrincipalPointOffsetFraction = options.calibration.maxSharedPrincipalPointOffsetFraction;
        compatibility.sharedFocalPriorSigma = options.calibration.sharedFocalPriorSigma;
        compatibility.sharedPrincipalPointPriorSigmaFraction =
            options.calibration.sharedPrincipalPointPriorSigmaFraction;
        compatibility.sharedFocalAspectPriorSigma = options.calibration.sharedFocalAspectPriorSigma;
        compatibility.maxSharedRadialK1Abs = options.calibration.maxSharedRadialK1Abs;
        compatibility.maxSharedRadialK2Abs = options.calibration.maxSharedRadialK2Abs;
        compatibility.maxSharedRadialK3Abs = options.calibration.maxSharedRadialK3Abs;
        compatibility.maxSharedTangentialP1Abs = options.calibration.maxSharedTangentialP1Abs;
        compatibility.maxSharedTangentialP2Abs = options.calibration.maxSharedTangentialP2Abs;
        compatibility.maxSharedSkewFraction = options.calibration.maxSharedSkewFraction;
        compatibility.maxSharedRadialK4Abs = options.calibration.maxSharedRadialK4Abs;
        compatibility.maxSharedTangentialP3Abs = options.calibration.maxSharedTangentialP3Abs;
        compatibility.maxSharedTangentialP4Abs = options.calibration.maxSharedTangentialP4Abs;
        compatibility.sharedRadialK1PriorSigma = options.calibration.sharedRadialK1PriorSigma;
        compatibility.sharedRadialK2PriorSigma = options.calibration.sharedRadialK2PriorSigma;
        compatibility.sharedRadialK3PriorSigma = options.calibration.sharedRadialK3PriorSigma;
        compatibility.sharedTangentialP1PriorSigma = options.calibration.sharedTangentialP1PriorSigma;
        compatibility.sharedTangentialP2PriorSigma = options.calibration.sharedTangentialP2PriorSigma;
        compatibility.sharedSkewPriorSigmaFraction = options.calibration.sharedSkewPriorSigmaFraction;
        compatibility.sharedRadialK4PriorSigma = options.calibration.sharedRadialK4PriorSigma;
        compatibility.sharedTangentialP3PriorSigma = options.calibration.sharedTangentialP3PriorSigma;
        compatibility.sharedTangentialP4PriorSigma = options.calibration.sharedTangentialP4PriorSigma;
        compatibility.sharedLowOrderDistortionScale = options.calibration.sharedLowOrderDistortionScale;
        compatibility.useReferenceCalibrationTransitionPrior =
            options.calibration.useReferenceCalibrationTransitionPrior;
        compatibility.referencePreviousIntrinsicParameterMask =
            options.calibration.referencePreviousIntrinsicParameterMask;

        compatibility.laserPlaneWeight = options.constraints.laserPlaneWeight;
        compatibility.laserHuberDeltaMeters = options.constraints.laserHuberDeltaMeters;
        compatibility.laserRangeWeight = options.constraints.laserRangeWeight;
        compatibility.laserRangeHuberDelta = options.constraints.laserRangeHuberDelta;
        compatibility.controlPointWeight = options.constraints.controlPointWeight;
        compatibility.controlPointHuberDeltaMeters = options.constraints.controlPointHuberDeltaMeters;
        compatibility.scaleBarWeight = options.constraints.scaleBarWeight;
        compatibility.scaleBarHuberDeltaMeters = options.constraints.scaleBarHuberDeltaMeters;
        compatibility.cameraPosePriorWeight = options.constraints.cameraPosePriorWeight;
        compatibility.cameraPosePriorHuberDelta = options.constraints.cameraPosePriorHuberDelta;
        compatibility.cameraPlaneHuberDelta = options.constraints.cameraPlaneHuberDelta;

        compatibility.backend = options.backend.requested;
        compatibility.plaMatrixDevice = options.backend.plaMatrixDevice;
        compatibility.enablePlaMatrixMixedPrecision = options.backend.enablePlaMatrixMixedPrecision;
        compatibility.minPlaMatrixCudaCameras = options.backend.minPlaMatrixCudaCameras;
        compatibility.minPlaMatrixCudaObservations = options.backend.minPlaMatrixCudaObservations;
        compatibility.minPlaMatrixVulkanCameras = options.backend.minPlaMatrixVulkanCameras;
        compatibility.minPlaMatrixVulkanObservations = options.backend.minPlaMatrixVulkanObservations;
        compatibility.minPlaMatrixOpenClCameras = options.backend.minPlaMatrixOpenClCameras;
        compatibility.minPlaMatrixOpenClObservations = options.backend.minPlaMatrixOpenClObservations;
        compatibility.minPlaMatrixDenseCameras = options.backend.minPlaMatrixDenseCameras;
        compatibility.minPlaMatrixCudaDenseObservations = options.backend.minPlaMatrixCudaDenseObservations;
        compatibility.minPlaMatrixVulkanDenseObservations = options.backend.minPlaMatrixVulkanDenseObservations;
        compatibility.minPlaMatrixOpenClDenseObservations = options.backend.minPlaMatrixOpenClDenseObservations;
        compatibility.plaMatrixPreconditionerClusterSize = options.backend.plaMatrixPreconditionerClusterSize;
        compatibility.allowBackendFallback = options.backend.allowFallback;

        compatibility.enableBackendQualityGate = options.quality.enabled;
        compatibility.maxAcceptedRmsGrowth = options.quality.maxAcceptedRmsGrowth;
        compatibility.minAcceptedValidTrackRatio = options.quality.minAcceptedValidTrackRatio;
        compatibility.maxAcceptedConstraintRmsGrowth = options.quality.maxAcceptedConstraintRmsGrowth;
        return compatibility;
    }

    SolveOptions makeSolveOptions(const Options& options)
    {
        SolveOptions structured;

        structured.solver.maxIterations = options.maxIterations;
        structured.solver.imageRobustLoss = options.imageRobustLoss;
        structured.solver.imageRobustLossScalePixels = options.imageRobustLossScalePixels;
        structured.solver.referenceArmijoCoefficient = options.referenceArmijoCoefficient;
        structured.solver.referenceLineSearchSteps = options.referenceLineSearchSteps;
        structured.solver.useReferenceOnlineSchur = options.useReferenceOnlineSchur;
        structured.solver.enablePointFilter = options.enablePointFilter;
        structured.solver.filterMaxReprojError = options.filterMaxReprojError;
        structured.solver.filterSigmaFactor = options.filterSigmaFactor;
        structured.solver.maxInitialTrackRms = options.maxInitialTrackRms;
        structured.solver.numThreads = options.numThreads;
        structured.solver.logIterationProgress = options.logIterationProgress;
        structured.solver.stopToken = options.stopToken;
        structured.solver.cancelFlag = options.cancelFlag;
        structured.solver.progressCallback = options.progressCallback;

        structured.calibration.refineCameraPose = options.refineCameraPose;
        structured.calibration.refineSharedFocalLength = options.refineSharedFocalLength;
        structured.calibration.refineSharedFocalAspectRatio = options.refineSharedFocalAspectRatio;
        structured.calibration.refineSharedPrincipalPoint = options.refineSharedPrincipalPoint;
        structured.calibration.refineSharedModelCoefficients = options.refineSharedModelCoefficients;
        structured.calibration.refineSharedRadialDistortion = options.refineSharedRadialDistortion;
        structured.calibration.refineSharedHighOrderDistortion = options.refineSharedHighOrderDistortion;
        structured.calibration.refineSharedMetashapeParameters = options.refineSharedMetashapeParameters;
        structured.calibration.hasTrustedSharedFocalPrior = options.hasTrustedSharedFocalPrior;
        structured.calibration.useSharedIntrinsicParameterMask = options.useSharedIntrinsicParameterMask;
        structured.calibration.sharedIntrinsicParameterMask = options.sharedIntrinsicParameterMask;
        structured.calibration.minSharedFocalScale = options.minSharedFocalScale;
        structured.calibration.maxSharedFocalScale = options.maxSharedFocalScale;
        structured.calibration.minSharedFocalAspectScale = options.minSharedFocalAspectScale;
        structured.calibration.maxSharedFocalAspectScale = options.maxSharedFocalAspectScale;
        structured.calibration.maxSharedPrincipalPointOffsetFraction = options.maxSharedPrincipalPointOffsetFraction;
        structured.calibration.sharedFocalPriorSigma = options.sharedFocalPriorSigma;
        structured.calibration.sharedPrincipalPointPriorSigmaFraction = options.sharedPrincipalPointPriorSigmaFraction;
        structured.calibration.sharedFocalAspectPriorSigma = options.sharedFocalAspectPriorSigma;
        structured.calibration.maxSharedRadialK1Abs = options.maxSharedRadialK1Abs;
        structured.calibration.maxSharedRadialK2Abs = options.maxSharedRadialK2Abs;
        structured.calibration.maxSharedRadialK3Abs = options.maxSharedRadialK3Abs;
        structured.calibration.maxSharedTangentialP1Abs = options.maxSharedTangentialP1Abs;
        structured.calibration.maxSharedTangentialP2Abs = options.maxSharedTangentialP2Abs;
        structured.calibration.maxSharedSkewFraction = options.maxSharedSkewFraction;
        structured.calibration.maxSharedRadialK4Abs = options.maxSharedRadialK4Abs;
        structured.calibration.maxSharedTangentialP3Abs = options.maxSharedTangentialP3Abs;
        structured.calibration.maxSharedTangentialP4Abs = options.maxSharedTangentialP4Abs;
        structured.calibration.sharedRadialK1PriorSigma = options.sharedRadialK1PriorSigma;
        structured.calibration.sharedRadialK2PriorSigma = options.sharedRadialK2PriorSigma;
        structured.calibration.sharedRadialK3PriorSigma = options.sharedRadialK3PriorSigma;
        structured.calibration.sharedTangentialP1PriorSigma = options.sharedTangentialP1PriorSigma;
        structured.calibration.sharedTangentialP2PriorSigma = options.sharedTangentialP2PriorSigma;
        structured.calibration.sharedSkewPriorSigmaFraction = options.sharedSkewPriorSigmaFraction;
        structured.calibration.sharedRadialK4PriorSigma = options.sharedRadialK4PriorSigma;
        structured.calibration.sharedTangentialP3PriorSigma = options.sharedTangentialP3PriorSigma;
        structured.calibration.sharedTangentialP4PriorSigma = options.sharedTangentialP4PriorSigma;
        structured.calibration.sharedLowOrderDistortionScale = options.sharedLowOrderDistortionScale;
        structured.calibration.useReferenceCalibrationTransitionPrior = options.useReferenceCalibrationTransitionPrior;
        structured.calibration.referencePreviousIntrinsicParameterMask =
            options.referencePreviousIntrinsicParameterMask;

        structured.constraints.laserPlaneWeight = options.laserPlaneWeight;
        structured.constraints.laserHuberDeltaMeters = options.laserHuberDeltaMeters;
        structured.constraints.laserRangeWeight = options.laserRangeWeight;
        structured.constraints.laserRangeHuberDelta = options.laserRangeHuberDelta;
        structured.constraints.controlPointWeight = options.controlPointWeight;
        structured.constraints.controlPointHuberDeltaMeters = options.controlPointHuberDeltaMeters;
        structured.constraints.scaleBarWeight = options.scaleBarWeight;
        structured.constraints.scaleBarHuberDeltaMeters = options.scaleBarHuberDeltaMeters;
        structured.constraints.cameraPosePriorWeight = options.cameraPosePriorWeight;
        structured.constraints.cameraPosePriorHuberDelta = options.cameraPosePriorHuberDelta;
        structured.constraints.cameraPlaneHuberDelta = options.cameraPlaneHuberDelta;

        structured.backend.requested = options.backend;
        structured.backend.plaMatrixDevice = options.plaMatrixDevice;
        structured.backend.enablePlaMatrixMixedPrecision = options.enablePlaMatrixMixedPrecision;
        structured.backend.minPlaMatrixCudaCameras = options.minPlaMatrixCudaCameras;
        structured.backend.minPlaMatrixCudaObservations = options.minPlaMatrixCudaObservations;
        structured.backend.minPlaMatrixVulkanCameras = options.minPlaMatrixVulkanCameras;
        structured.backend.minPlaMatrixVulkanObservations = options.minPlaMatrixVulkanObservations;
        structured.backend.minPlaMatrixOpenClCameras = options.minPlaMatrixOpenClCameras;
        structured.backend.minPlaMatrixOpenClObservations = options.minPlaMatrixOpenClObservations;
        structured.backend.minPlaMatrixDenseCameras = options.minPlaMatrixDenseCameras;
        structured.backend.minPlaMatrixCudaDenseObservations = options.minPlaMatrixCudaDenseObservations;
        structured.backend.minPlaMatrixVulkanDenseObservations = options.minPlaMatrixVulkanDenseObservations;
        structured.backend.minPlaMatrixOpenClDenseObservations = options.minPlaMatrixOpenClDenseObservations;
        structured.backend.plaMatrixPreconditionerClusterSize = options.plaMatrixPreconditionerClusterSize;
        structured.backend.allowFallback = options.allowBackendFallback;

        structured.quality.enabled = options.enableBackendQualityGate;
        structured.quality.maxAcceptedRmsGrowth = options.maxAcceptedRmsGrowth;
        structured.quality.minAcceptedValidTrackRatio = options.minAcceptedValidTrackRatio;
        structured.quality.maxAcceptedConstraintRmsGrowth = options.maxAcceptedConstraintRmsGrowth;
        return structured;
    }

    bool validateOptions(const SolveOptions& options, std::string* error) noexcept
    {
        try
        {
            return validateOptions(makeCompatibilityOptions(options), error);
        }
        catch (...)
        {
            setError(error, "failed to materialize structured solver options");
            return false;
        }
    }

    bool validateOptions(const Options& options, std::string* error) noexcept
    {
        if (error)
        {
            try
            {
                error->clear();
            }
            catch (...)
            {
            }
        }
        switch (options.backend)
        {
        case Backend::Auto:
        case Backend::PlaMatrixCpu:
        case Backend::PlaMatrixCuda:
        case Backend::PlaMatrixVulkan:
        case Backend::PlaMatrixOpenCl:
            break;
        default:
            setError(error, "backend value is invalid");
            return false;
        }
        if (options.maxIterations <= 0)
        {
            setError(error, "maximum iteration count must be positive");
            return false;
        }
        switch (options.imageRobustLoss)
        {
        case ImageRobustLoss::LeastSquares:
        case ImageRobustLoss::Huber:
        case ImageRobustLoss::Cauchy:
            break;
        default:
            setError(error, "image robust loss value is invalid");
            return false;
        }
        if (!positive(options.imageRobustLossScalePixels))
        {
            setError(error, "image robust loss scale must be finite and positive");
            return false;
        }
        if (!positive(options.minSharedFocalScale) || !finite(options.maxSharedFocalScale) ||
            options.maxSharedFocalScale < options.minSharedFocalScale)
        {
            setError(error, "shared focal scale bounds must be finite, positive, and ordered");
            return false;
        }
        if (!positive(options.minSharedFocalAspectScale) || !finite(options.maxSharedFocalAspectScale) ||
            options.maxSharedFocalAspectScale < options.minSharedFocalAspectScale)
        {
            setError(error, "shared focal-aspect bounds must be finite, positive, and ordered");
            return false;
        }
        if (!positive(options.maxSharedPrincipalPointOffsetFraction) || !positive(options.sharedFocalPriorSigma) ||
            !positive(options.sharedPrincipalPointPriorSigmaFraction) || !positive(options.sharedFocalAspectPriorSigma))
        {
            setError(error, "shared intrinsic offsets and prior sigmas are invalid");
            return false;
        }
        const std::array<double, 9> distortion_bounds{{options.maxSharedRadialK1Abs,
                                                       options.maxSharedRadialK2Abs,
                                                       options.maxSharedRadialK3Abs,
                                                       options.maxSharedTangentialP1Abs,
                                                       options.maxSharedTangentialP2Abs,
                                                       options.maxSharedSkewFraction,
                                                       options.maxSharedRadialK4Abs,
                                                       options.maxSharedTangentialP3Abs,
                                                       options.maxSharedTangentialP4Abs}};
        if (!std::all_of(distortion_bounds.begin(), distortion_bounds.end(), positive))
        {
            setError(error, "shared distortion bounds must be finite and positive");
            return false;
        }
        const std::array<double, 9> distortion_sigmas{{options.sharedRadialK1PriorSigma,
                                                       options.sharedRadialK2PriorSigma,
                                                       options.sharedRadialK3PriorSigma,
                                                       options.sharedTangentialP1PriorSigma,
                                                       options.sharedTangentialP2PriorSigma,
                                                       options.sharedSkewPriorSigmaFraction,
                                                       options.sharedRadialK4PriorSigma,
                                                       options.sharedTangentialP3PriorSigma,
                                                       options.sharedTangentialP4PriorSigma}};
        if (!std::all_of(distortion_sigmas.begin(), distortion_sigmas.end(), positive) ||
            !finite(options.sharedLowOrderDistortionScale) || options.sharedLowOrderDistortionScale < 1.0)
        {
            setError(error,
                     "shared distortion prior sigmas must be positive and the low-order scale must be at least 1");
            return false;
        }
        if (!finite(options.referenceArmijoCoefficient) || options.referenceArmijoCoefficient <= 0.0 ||
            options.referenceArmijoCoefficient >= 1.0 || options.referenceLineSearchSteps < 1 ||
            options.referenceLineSearchSteps > 64)
        {
            setError(error, "Armijo coefficient/line-search step count is invalid");
            return false;
        }
        const std::array<double, 5> constraint_weights{{options.laserPlaneWeight,
                                                        options.laserRangeWeight,
                                                        options.controlPointWeight,
                                                        options.scaleBarWeight,
                                                        options.cameraPosePriorWeight}};
        if (!std::all_of(constraint_weights.begin(), constraint_weights.end(), positive))
        {
            setError(error, "constraint weights must be finite and positive");
            return false;
        }
        const std::array<double, 6> robust_thresholds{{options.laserHuberDeltaMeters,
                                                       options.laserRangeHuberDelta,
                                                       options.controlPointHuberDeltaMeters,
                                                       options.scaleBarHuberDeltaMeters,
                                                       options.cameraPosePriorHuberDelta,
                                                       options.cameraPlaneHuberDelta}};
        if (!std::all_of(robust_thresholds.begin(), robust_thresholds.end(), nonNegative))
        {
            setError(error, "constraint robust thresholds must be finite and non-negative");
            return false;
        }
        if (!nonNegative(options.filterMaxReprojError) || !nonNegative(options.filterSigmaFactor) ||
            options.numThreads < 0 || options.plaMatrixDevice < 0)
        {
            setError(error, "filter, thread, or device options are invalid");
            return false;
        }
        const std::array<int, 10> scale_thresholds{{options.minPlaMatrixCudaCameras,
                                                    options.minPlaMatrixCudaObservations,
                                                    options.minPlaMatrixVulkanCameras,
                                                    options.minPlaMatrixVulkanObservations,
                                                    options.minPlaMatrixOpenClCameras,
                                                    options.minPlaMatrixOpenClObservations,
                                                    options.minPlaMatrixDenseCameras,
                                                    options.minPlaMatrixCudaDenseObservations,
                                                    options.minPlaMatrixVulkanDenseObservations,
                                                    options.minPlaMatrixOpenClDenseObservations}};
        if (std::any_of(scale_thresholds.begin(), scale_thresholds.end(), [](int value) { return value < 0; }))
        {
            setError(error, "automatic backend scale thresholds must be non-negative");
            return false;
        }
        if (!nonNegative(options.maxInitialTrackRms) || options.plaMatrixPreconditionerClusterSize < 1 ||
            options.plaMatrixPreconditionerClusterSize > 16 || !finite(options.maxAcceptedRmsGrowth) ||
            options.maxAcceptedRmsGrowth < 1.0 || !finite(options.minAcceptedValidTrackRatio) ||
            options.minAcceptedValidTrackRatio < 0.0 || options.minAcceptedValidTrackRatio > 1.0 ||
            !finite(options.maxAcceptedConstraintRmsGrowth) || options.maxAcceptedConstraintRmsGrowth < 1.0)
        {
            setError(error, "backend quality-gate or preconditioner options are invalid");
            return false;
        }
        return true;
    }

} // namespace plabundle
