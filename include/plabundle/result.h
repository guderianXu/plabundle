#pragma once

#include <plabundle/backend.h>
#include <plabundle/camera.h>
#include <plabundle/constraints.h>
#include <plabundle/options.h>
#include <plabundle/rig.h>

#include <array>
#include <string>
#include <vector>

namespace plabundle
{

    struct RefinedPoint
    {
        bool valid = false;
        bool converged = false;
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        double rmsBefore = 0.0;
        double rmsAfter = 0.0;
        int iterations = 0;
    };

    struct RefinedLaserRangeShot
    {
        bool valid = false;
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        LaserPointMode pointMode = LaserPointMode::Unspecified;
        std::string shotId;
        double ephemerisTimeSeconds = 0.0;
        int sourceIndex = -1;
        double computedRangeBeforeMeters = 0.0;
        double computedRangeAfterMeters = 0.0;
        double residualBeforeMeters = 0.0;
        double residualAfterMeters = 0.0;
        double normalizedResidualAfter = 0.0;
    };

    struct TimingSummary
    {
        double setupSeconds = 0.0;
        double solveSeconds = 0.0;
        double postprocessSeconds = 0.0;
        double totalSeconds = 0.0;
    };

    struct QualitySummary
    {
        int totalTracks = 0;
        int optimizedTracks = 0;
        double validTrackRatio = 0.0;
        double meanRmsBefore = 0.0;
        double meanRmsAfter = 0.0;
        int refinedCameraCount = 0;
        int refinedIntrinsicCount = 0;
        int refinedCalibrationGroupCount = 0;
        int selfCalibrationStagesRun = 0;
        double refinedSharedFocalScale = 1.0;
        double refinedSharedFocalAspectScale = 1.0;
        double refinedSharedPrincipalOffsetX = 0.0;
        double refinedSharedPrincipalOffsetY = 0.0;
        double refinedSharedRadialK1 = 0.0;
        double refinedSharedRadialK2 = 0.0;
        double refinedSharedRadialK3 = 0.0;
        double refinedSharedTangentialP1 = 0.0;
        double refinedSharedTangentialP2 = 0.0;
        double refinedSharedSkewB2 = 0.0;
        double refinedSharedRadialK4 = 0.0;
        double refinedSharedTangentialP3 = 0.0;
        double refinedSharedTangentialP4 = 0.0;
        IntrinsicParameterMask referenceCommittedIntrinsicParameterMask{};
        int laserConstraintCount = 0;
        double laserRmsBeforeMeters = 0.0;
        double laserRmsAfterMeters = 0.0;
        double laserMedianBeforeMeters = 0.0;
        double laserMedianAfterMeters = 0.0;
        int laserRangeConstraintCount = 0;
        double laserRangeRmsBeforeMeters = 0.0;
        double laserRangeRmsAfterMeters = 0.0;
        int controlPointConstraintCount = 0;
        double controlPointRmsBeforeMeters = 0.0;
        double controlPointRmsAfterMeters = 0.0;
        int scaleBarConstraintCount = 0;
        double scaleBarRmsBeforeMeters = 0.0;
        double scaleBarRmsAfterMeters = 0.0;
    };

    struct PlaMatrixDiagnostics
    {
        double initialCost = 0.0;
        double finalCost = 0.0;
        int acceptedSteps = 0;
        int rejectedSteps = 0;
        int linearizations = 0;
        int objectiveEvaluations = 0;
        int rejectedInitialTracks = 0;
        bool referenceOnlineSchurUsed = false;
        std::string linearSolverName = "none";
        std::string preconditionerName = "none";
        std::string deviceName;
        int linearIterations = 0;
        int denseFallbacks = 0;
        std::string denseFallbackMessage;
        double assemblySeconds = 0.0;
        double objectiveSeconds = 0.0;
        double trialStateSeconds = 0.0;
        double linearToleranceMinimum = 0.0;
        double linearToleranceMaximum = 0.0;
        int schurPatternBuilds = 0;
        int schurPatternReuses = 0;
        bool schurAssemblyOnDevice = false;
        // True only when a requested FP32 PCG seed converged and was accepted before the FP64 solve.
        bool mixedPrecisionUsed = false;
        double smallBlockInverseSeconds = 0.0;
        double schurAccumulationSeconds = 0.0;
        double csrConversionSeconds = 0.0;
        double schurAssemblySeconds = 0.0;
        double choleskyFactorizationSeconds = 0.0;
        double triangularSolveSeconds = 0.0;
        double symbolicAnalysisSeconds = 0.0;
        int symbolicAnalysisReuses = 0;
        double residualCheckSeconds = 0.0;
        double linearSolveSeconds = 0.0;
        double backSubstitutionSeconds = 0.0;
    };

    struct Result
    {
        Backend requestedBackend = Backend::PlaMatrixCpu;
        Backend usedBackend = Backend::PlaMatrixCpu;
        SolveStatus status = SolveStatus::NotRun;
        bool solutionUsable = false;
        bool usedGpu = false;
        bool backendFallback = false;
        std::string backendMessage;
        std::string backendSelectionReason;
        bool qualityGateRejected = false;
        std::string qualityGateMessage;
        int observationCount = 0;
        TimingSummary timing;
        QualitySummary quality;
        PlaMatrixDiagnostics plaMatrix;
        std::vector<RefinedPoint> points;
        std::vector<RefinedLaserRangeShot> laserRangeShots;
        std::vector<FrameCamera> refinedCameras;
        RigTopology refinedRig;

        bool usable() const noexcept
        {
            return solutionUsable && (status == SolveStatus::Success || status == SolveStatus::NoConvergence);
        }
    };

} // namespace plabundle
