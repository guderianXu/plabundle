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
            static_cast<CameraPosePrior&>(target) = source;
            target.enabled = true;
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
        if (target.cameraCalibrationGroupIds.empty() && !problem.rig.empty())
        {
            target.cameraCalibrationGroupIds.assign(problem.cameras.size(), 0);
            std::vector<std::pair<int, int>> sensor_keys;
            for (const placamera::RigCameraBinding& binding : problem.rig.cameraBindings)
            {
                const std::pair<int, int> key{binding.rigId, binding.sensorId};
                auto found = std::find(sensor_keys.begin(), sensor_keys.end(), key);
                if (found == sensor_keys.end())
                {
                    found = sensor_keys.insert(sensor_keys.end(), key);
                }
                target.cameraCalibrationGroupIds[static_cast<std::size_t>(binding.cameraIndex)] =
                    static_cast<int>(std::distance(sensor_keys.begin(), found));
            }
        }
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
        if (!problem.rig.empty())
        {
            for (const placamera::RigCameraBinding& binding : problem.rig.cameraBindings)
            {
                const auto capture =
                    std::find_if(problem.rig.captures.begin(),
                                 problem.rig.captures.end(),
                                 [&](const placamera::RigCapture& value)
                                 { return value.rigId == binding.rigId && value.captureId == binding.captureId; });
                const auto sensor =
                    std::find_if(problem.rig.sensors.begin(),
                                 problem.rig.sensors.end(),
                                 [&](const placamera::RigSensor& value)
                                 { return value.rigId == binding.rigId && value.sensorId == binding.sensorId; });
                if (capture->fixedPose && sensor->fixedExtrinsic &&
                    std::find(target.fixedCameraIndices.begin(),
                              target.fixedCameraIndices.end(),
                              binding.cameraIndex) == target.fixedCameraIndices.end())
                {
                    target.fixedCameraIndices.push_back(binding.cameraIndex);
                }
            }
        }
        target.fixedTrackIndices = problem.fixedTrackIndices;
        target.rig = problem.rig;
        return target;
    }

    Result makePublicResult(const BAResult& source,
                            const Problem& problem,
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
        target.quality.refinedSharedSkewB2 = source.refinedSharedSkewB2;
        target.quality.refinedSharedRadialK4 = source.refinedSharedRadialK4;
        target.quality.refinedSharedTangentialP3 = source.refinedSharedTangentialP3;
        target.quality.refinedSharedTangentialP4 = source.refinedSharedTangentialP4;
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
        target.refinedCameras = makeNumericStates(source.refinedCameras, problem.cameras);
        if (target.refinedCameras.size() != source.refinedCameras.size())
        {
            target.solutionUsable = false;
            target.backendMessage += "; refined_camera_metadata_mismatch";
        }
        target.refinedRig = source.refinedRig;
        return target;
    }

} // namespace plabundle::internal
