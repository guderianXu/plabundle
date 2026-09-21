#include "Conversion.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>

namespace plabundle::internal
{
    namespace
    {
        bool hasLaserPlaneConstraints(const Problem& problem)
        {
            return std::any_of(problem.tracks.begin(),
                               problem.tracks.end(),
                               [](const Track& track) { return !track.laserPlaneConstraints.empty(); });
        }

        bool hasControlPointConstraints(const Problem& problem)
        {
            return std::any_of(problem.tracks.begin(),
                               problem.tracks.end(),
                               [](const Track& track) { return !track.controlPointConstraints.empty(); });
        }

        BACameraPosePrior makePosePrior(const CameraPosePrior& source)
        {
            BACameraPosePrior target;
            target.enabled = true;
            target.cameraToWorldRotation = source.cameraToWorldRotation;
            target.cameraCenter = source.cameraCenter;
            target.positionSigmaMeters = source.positionSigmaMeters;
            target.rotationSigmaDegrees = source.rotationSigmaDegrees;
            return target;
        }

        BACameraPlaneConstraint makeCameraPlaneConstraint(const CameraPlaneConstraint& source)
        {
            BACameraPlaneConstraint target;
            target.enabled = true;
            target.point = source.point;
            target.normal = source.normal;
            target.referenceSignedDistances = source.referenceSignedDistances;
            target.sigmaMeters = source.sigmaMeters;
            target.weight = source.weight;
            return target;
        }

        RefinedPoint makeRefinedPoint(const BARefinedPoint& source)
        {
            RefinedPoint target;
            target.valid = source.valid;
            target.converged = source.converged;
            target.point = source.point;
            target.rmsBefore = source.rmsBefore;
            target.rmsAfter = source.rmsAfter;
            target.iterations = source.iterations;
            return target;
        }

        RefinedLaserRangeShot makeRefinedLaserRangeShot(const BARefinedLaserRangeShot& source)
        {
            RefinedLaserRangeShot target;
            target.valid = source.valid;
            target.point = source.point;
            target.pointMode = source.pointMode;
            target.shotId = source.shotId;
            target.ephemerisTimeSeconds = source.ephemerisTimeSeconds;
            target.sourceIndex = source.sourceIndex;
            target.computedRangeBeforeMeters = source.computedRangeBeforeMeters;
            target.computedRangeAfterMeters = source.computedRangeAfterMeters;
            target.residualBeforeMeters = source.residualBeforeMeters;
            target.residualAfterMeters = source.residualAfterMeters;
            target.normalizedResidualAfter = source.normalizedResidualAfter;
            return target;
        }

    } // namespace

    BAOptions makeSolverOptions(const Problem& problem, const Options& options, Backend usedBackend)
    {
        BAOptions target;
        static_cast<Options&>(target) = options;
        target.backend = usedBackend;
        target.cameraCalibrationGroupIds = problem.cameraCalibrationGroupIds;
        target.sharedIntrinsicReferenceCameras = makeCameraStates(problem.sharedIntrinsicReferenceCameras);
        target.referenceGaugeAnchorCameraIndex = problem.gauge.referenceAnchorCameraIndex;
        target.referenceGaugeScaleCameraIndex = problem.gauge.referenceScaleCameraIndex;
        target.referenceGaugeBaseline = problem.gauge.referenceBaseline;
        target.enableLaserPlaneConstraints = hasLaserPlaneConstraints(problem);
        target.enableLaserRangeConstraints = !problem.laserRangeConstraints.empty();
        target.laserRangeConstraints = problem.laserRangeConstraints;
        target.enableControlPointConstraints = hasControlPointConstraints(problem);
        target.enableScaleBarConstraints = !problem.scaleBarConstraints.empty();
        target.scaleBarConstraints = problem.scaleBarConstraints;
        target.cameraPosePriors.reserve(problem.cameraPosePriors.size());
        for (const std::optional<CameraPosePrior>& prior : problem.cameraPosePriors)
        {
            target.cameraPosePriors.push_back(prior ? makePosePrior(*prior) : BACameraPosePrior{});
        }
        if (problem.cameraPlaneConstraint)
        {
            target.cameraPlaneConstraint = makeCameraPlaneConstraint(*problem.cameraPlaneConstraint);
        }
        target.gaugePolicy = problem.gauge.policy;
        target.fixedCameraIndices = problem.fixedCameraIndices;
        target.fixedTrackIndices = problem.fixedTrackIndices;
        return target;
    }

    Result makePublicResult(const BAResult& source,
                            Backend requestedBackend,
                            Backend usedBackend,
                            const std::string& selectionReason)
    {
        Result target;
        target.requestedBackend = requestedBackend;
        target.usedBackend = usedBackend;
        target.status = source.solveStatus;
        target.solutionUsable = source.solutionUsable;
        target.usedGpu = source.usedGpu;
        target.backendFallback = source.backendFallback;
        target.backendMessage = source.backendMessage;
        target.backendSelectionReason = selectionReason.empty() ? source.backendSelectionReason : selectionReason;
        target.qualityGateRejected = source.qualityGateRejected;
        target.qualityGateMessage = source.qualityGateMessage;
        target.observationCount = source.observationCount;

        target.timing.setupSeconds = source.setupSeconds;
        target.timing.solveSeconds = source.solveSeconds;
        target.timing.postprocessSeconds = source.postprocessSeconds;
        target.timing.totalSeconds = source.totalSeconds;

        target.quality.totalTracks = source.totalTracks;
        target.quality.optimizedTracks = source.optimizedTracks;
        target.quality.validTrackRatio = source.validTrackRatio;
        target.quality.meanRmsBefore = source.meanRmsBefore;
        target.quality.meanRmsAfter = source.meanRmsAfter;
        target.quality.refinedCameraCount = source.refinedCameraCount;
        target.quality.refinedIntrinsicCount = source.refinedIntrinsicCount;
        target.quality.refinedCalibrationGroupCount = source.refinedCalibrationGroupCount;
        target.quality.selfCalibrationStagesRun = source.selfCalibrationStagesRun;
        target.quality.refinedSharedFocalScale = source.refinedSharedFocalScale;
        target.quality.refinedSharedFocalAspectScale = source.refinedSharedFocalAspectScale;
        target.quality.refinedSharedPrincipalOffsetX = source.refinedSharedPrincipalOffsetX;
        target.quality.refinedSharedPrincipalOffsetY = source.refinedSharedPrincipalOffsetY;
        target.quality.refinedSharedRadialK1 = source.refinedSharedRadialK1;
        target.quality.refinedSharedRadialK2 = source.refinedSharedRadialK2;
        target.quality.refinedSharedRadialK3 = source.refinedSharedRadialK3;
        target.quality.refinedSharedTangentialP1 = source.refinedSharedTangentialP1;
        target.quality.refinedSharedTangentialP2 = source.refinedSharedTangentialP2;
        target.quality.referenceCommittedIntrinsicParameterMask = source.referenceCommittedIntrinsicParameterMask;
        target.quality.laserConstraintCount = source.laserConstraintCount;
        target.quality.laserRmsBeforeMeters = source.laserRmsBeforeMeters;
        target.quality.laserRmsAfterMeters = source.laserRmsAfterMeters;
        target.quality.laserMedianBeforeMeters = source.laserMedianBeforeMeters;
        target.quality.laserMedianAfterMeters = source.laserMedianAfterMeters;
        target.quality.laserRangeConstraintCount = source.laserRangeConstraintCount;
        target.quality.laserRangeRmsBeforeMeters = source.laserRangeRmsBeforeMeters;
        target.quality.laserRangeRmsAfterMeters = source.laserRangeRmsAfterMeters;
        target.quality.controlPointConstraintCount = source.controlPointConstraintCount;
        target.quality.controlPointRmsBeforeMeters = source.controlPointRmsBeforeMeters;
        target.quality.controlPointRmsAfterMeters = source.controlPointRmsAfterMeters;
        target.quality.scaleBarConstraintCount = source.scaleBarConstraintCount;
        target.quality.scaleBarRmsBeforeMeters = source.scaleBarRmsBeforeMeters;
        target.quality.scaleBarRmsAfterMeters = source.scaleBarRmsAfterMeters;

        target.plaMatrix.initialCost = source.plaMatrixInitialCost;
        target.plaMatrix.finalCost = source.plaMatrixFinalCost;
        target.plaMatrix.acceptedSteps = source.plaMatrixAcceptedSteps;
        target.plaMatrix.rejectedSteps = source.plaMatrixRejectedSteps;
        target.plaMatrix.linearizations = source.plaMatrixLinearizations;
        target.plaMatrix.objectiveEvaluations = source.plaMatrixObjectiveEvaluations;
        target.plaMatrix.rejectedInitialTracks = source.plaMatrixRejectedInitialTracks;
        target.plaMatrix.referenceOnlineSchurUsed = source.plaMatrixReferenceOnlineSchurUsed;
        target.plaMatrix.linearSolverName = source.plaMatrixLinearSolverName;
        target.plaMatrix.preconditionerName = source.plaMatrixPreconditionerName;
        target.plaMatrix.deviceName = source.plaMatrixDeviceName;
        target.plaMatrix.linearIterations = source.plaMatrixLinearIterations;
        target.plaMatrix.denseFallbacks = source.plaMatrixDenseFallbacks;
        target.plaMatrix.denseFallbackMessage = source.plaMatrixDenseFallbackMessage;
        target.plaMatrix.assemblySeconds = source.plaMatrixAssemblySeconds;
        target.plaMatrix.objectiveSeconds = source.plaMatrixObjectiveSeconds;
        target.plaMatrix.trialStateSeconds = source.plaMatrixTrialStateSeconds;
        target.plaMatrix.linearToleranceMinimum = source.plaMatrixLinearToleranceMinimum;
        target.plaMatrix.linearToleranceMaximum = source.plaMatrixLinearToleranceMaximum;
        target.plaMatrix.schurPatternBuilds = source.plaMatrixSchurPatternBuilds;
        target.plaMatrix.schurPatternReuses = source.plaMatrixSchurPatternReuses;
        target.plaMatrix.schurAssemblyOnDevice = source.plaMatrixSchurAssemblyOnDevice;
        target.plaMatrix.mixedPrecisionUsed = source.plaMatrixMixedPrecisionUsed;
        target.plaMatrix.smallBlockInverseSeconds = source.plaMatrixSmallBlockInverseSeconds;
        target.plaMatrix.schurAccumulationSeconds = source.plaMatrixSchurAccumulationSeconds;
        target.plaMatrix.csrConversionSeconds = source.plaMatrixCsrConversionSeconds;
        target.plaMatrix.schurAssemblySeconds = source.plaMatrixSchurAssemblySeconds;
        target.plaMatrix.choleskyFactorizationSeconds = source.plaMatrixCholeskyFactorizationSeconds;
        target.plaMatrix.triangularSolveSeconds = source.plaMatrixTriangularSolveSeconds;
        target.plaMatrix.symbolicAnalysisSeconds = source.plaMatrixSymbolicAnalysisSeconds;
        target.plaMatrix.symbolicAnalysisReuses = source.plaMatrixSymbolicAnalysisReuses;
        target.plaMatrix.residualCheckSeconds = source.plaMatrixResidualCheckSeconds;
        target.plaMatrix.linearSolveSeconds = source.plaMatrixLinearSolveSeconds;
        target.plaMatrix.backSubstitutionSeconds = source.plaMatrixBackSubstitutionSeconds;

        target.points.reserve(source.points.size());
        std::transform(source.points.begin(), source.points.end(), std::back_inserter(target.points), makeRefinedPoint);
        target.laserRangeShots.reserve(source.laserRangeShots.size());
        std::transform(source.laserRangeShots.begin(),
                       source.laserRangeShots.end(),
                       std::back_inserter(target.laserRangeShots),
                       makeRefinedLaserRangeShot);
        target.refinedCameras = makeFrameCameras(source.refinedCameras);
        return target;
    }

} // namespace plabundle::internal
