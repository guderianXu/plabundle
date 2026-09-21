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
        const std::array<double, 5> distortion_bounds{{options.maxSharedRadialK1Abs,
                                                       options.maxSharedRadialK2Abs,
                                                       options.maxSharedRadialK3Abs,
                                                       options.maxSharedTangentialP1Abs,
                                                       options.maxSharedTangentialP2Abs}};
        if (!std::all_of(distortion_bounds.begin(), distortion_bounds.end(), positive))
        {
            setError(error, "shared distortion bounds must be finite and positive");
            return false;
        }
        const std::array<double, 5> distortion_sigmas{{options.sharedRadialK1PriorSigma,
                                                       options.sharedRadialK2PriorSigma,
                                                       options.sharedRadialK3PriorSigma,
                                                       options.sharedTangentialP1PriorSigma,
                                                       options.sharedTangentialP2PriorSigma}};
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
        const std::array<int, 7> scale_thresholds{{options.minPlaMatrixCudaCameras,
                                                   options.minPlaMatrixCudaObservations,
                                                   options.minPlaMatrixOpenClCameras,
                                                   options.minPlaMatrixOpenClObservations,
                                                   options.minPlaMatrixDenseCameras,
                                                   options.minPlaMatrixCudaDenseObservations,
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
