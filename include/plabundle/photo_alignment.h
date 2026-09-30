#pragma once

#include <plabundle/problem.h>
#include <plabundle/result.h>

#include <string>
#include <vector>

namespace plabundle
{

    /**
     * Associates PlaBundle's dense indices with stable caller-owned identifiers.
     *
     * The identifiers may come from a project database, an exchange format, or a
     * legacy solver. PlaBundle deliberately does not interpret their syntax.
     */
    struct PhotoAlignmentProblem
    {
        Problem problem;
        std::vector<std::string> cameraIds;
        std::vector<std::string> trackIds;
    };

    bool validatePhotoAlignmentProblem(const PhotoAlignmentProblem& problem, std::string* error = nullptr);

    struct PhotoAlignmentCameraState
    {
        std::string id;
        placamera::FramePinholeNumericState camera;
    };

    struct PhotoAlignmentTrackState
    {
        std::string id;
        RefinedPoint point;
    };

    /** A solver-neutral snapshot suitable for legacy/new implementation comparisons. */
    struct PhotoAlignmentOutcome
    {
        SolveStatus terminationStatus = SolveStatus::NotRun;
        std::string terminationReason;
        Backend requestedBackend = Backend::PlaMatrixCpu;
        Backend usedBackend = Backend::PlaMatrixCpu;
        std::string backendSelectionReason;
        bool solutionUsable = false;
        bool backendFallback = false;
        bool qualityGateRejected = false;
        double meanRmsBefore = 0.0;
        double meanRmsAfter = 0.0;
        std::vector<PhotoAlignmentCameraState> cameras;
        std::vector<PhotoAlignmentTrackState> tracks;
    };

    bool makePhotoAlignmentOutcome(const PhotoAlignmentProblem& problem,
                                   const Result& result,
                                   PhotoAlignmentOutcome* outcome,
                                   std::string* error = nullptr);

    bool validatePhotoAlignmentOutcome(const PhotoAlignmentOutcome& outcome, std::string* error = nullptr);

    struct PhotoAlignmentComparisonTolerance
    {
        double rms = 1.0e-8;
        double cameraCenterMeters = 1.0e-8;
        double cameraRotationDegrees = 1.0e-8;
        double pointMeters = 1.0e-8;
        bool requireTerminationStatusMatch = true;
        bool requireTerminationReasonMatch = true;
        bool requireRequestedBackendMatch = true;
        bool requireUsedBackendMatch = true;
    };

    struct PhotoAlignmentCameraDifference
    {
        std::string id;
        double centerDistanceMeters = 0.0;
        double rotationAngleDegrees = 0.0;
    };

    struct PhotoAlignmentTrackDifference
    {
        std::string id;
        bool referenceValid = false;
        bool candidateValid = false;
        double pointDistanceMeters = 0.0;
    };

    struct PhotoAlignmentComparison
    {
        bool comparable = false;
        bool equivalent = false;
        bool terminationStatusMatches = false;
        bool terminationReasonMatches = false;
        bool requestedBackendMatches = false;
        bool usedBackendMatches = false;
        double rmsBeforeDifference = 0.0;
        double rmsAfterDifference = 0.0;
        double maximumCameraCenterDifferenceMeters = 0.0;
        double maximumCameraRotationDifferenceDegrees = 0.0;
        double maximumPointDifferenceMeters = 0.0;
        std::vector<std::string> validTrackMaskMismatchIds;
        std::vector<PhotoAlignmentCameraDifference> cameraDifferences;
        std::vector<PhotoAlignmentTrackDifference> trackDifferences;
        std::string message;
    };

    bool comparePhotoAlignmentOutcomes(const PhotoAlignmentOutcome& reference,
                                       const PhotoAlignmentOutcome& candidate,
                                       const PhotoAlignmentComparisonTolerance& tolerance,
                                       PhotoAlignmentComparison* comparison,
                                       std::string* error = nullptr);

} // namespace plabundle
