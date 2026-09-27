#include <plabundle/photo_alignment.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace plabundle
{
    namespace
    {
        constexpr double kRadiansToDegrees = 57.2957795130823208768;

        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        bool validIds(const std::vector<std::string>& ids, const char* label, std::string* error)
        {
            std::set<std::string> unique;
            for (const std::string& id : ids)
            {
                if (id.empty())
                {
                    setError(error, std::string(label) + " ids must not be empty");
                    return false;
                }
                if (!unique.insert(id).second)
                {
                    setError(error, std::string(label) + " ids must be unique: " + id);
                    return false;
                }
            }
            return true;
        }

        bool validTolerance(const PhotoAlignmentComparisonTolerance& tolerance, std::string* error)
        {
            if (!finite(tolerance.rms) || tolerance.rms < 0.0 || !finite(tolerance.cameraCenterMeters) ||
                tolerance.cameraCenterMeters < 0.0 || !finite(tolerance.cameraRotationDegrees) ||
                tolerance.cameraRotationDegrees < 0.0 || !finite(tolerance.pointMeters) || tolerance.pointMeters < 0.0)
            {
                setError(error, "photo-alignment comparison tolerances must be finite and non-negative");
                return false;
            }
            return true;
        }

        double distance(const std::array<double, 3>& left, const std::array<double, 3>& right) noexcept
        {
            return std::hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
        }

        double rotationDifferenceDegrees(const std::array<double, 9>& left, const std::array<double, 9>& right) noexcept
        {
            double trace = 0.0;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    trace += left[static_cast<std::size_t>(column * 3 + row)] *
                             right[static_cast<std::size_t>(column * 3 + row)];
                }
            }
            const double cosine = std::clamp(0.5 * (trace - 1.0), -1.0, 1.0);
            return std::acos(cosine) * kRadiansToDegrees;
        }

        template <typename State>
        bool makeIndex(const std::vector<State>& states,
                       const char* label,
                       std::map<std::string, const State*>* index,
                       std::string* error)
        {
            index->clear();
            for (const State& state : states)
            {
                if (state.id.empty())
                {
                    setError(error, std::string(label) + " id must not be empty");
                    return false;
                }
                if (!index->emplace(state.id, &state).second)
                {
                    setError(error, std::string(label) + " ids must be unique: " + state.id);
                    return false;
                }
            }
            return true;
        }

        template <typename State>
        bool sameIds(const std::map<std::string, const State*>& reference,
                     const std::map<std::string, const State*>& candidate) noexcept
        {
            if (reference.size() != candidate.size())
            {
                return false;
            }
            return std::equal(reference.begin(),
                              reference.end(),
                              candidate.begin(),
                              [](const auto& left, const auto& right) { return left.first == right.first; });
        }

    } // namespace

    bool validatePhotoAlignmentProblem(const PhotoAlignmentProblem& problem, std::string* error)
    {
        if (problem.cameraIds.size() != problem.problem.cameras.size())
        {
            setError(error, "photo-alignment camera ids must align with problem cameras");
            return false;
        }
        if (problem.trackIds.size() != problem.problem.tracks.size())
        {
            setError(error, "photo-alignment track ids must align with problem tracks");
            return false;
        }
        return validIds(problem.cameraIds, "camera", error) && validIds(problem.trackIds, "track", error) &&
               validateProblem(problem.problem, error);
    }

    bool makePhotoAlignmentOutcome(const PhotoAlignmentProblem& problem,
                                   const Result& result,
                                   PhotoAlignmentOutcome* outcome,
                                   std::string* error)
    {
        if (!outcome)
        {
            setError(error, "photo-alignment outcome output is null");
            return false;
        }
        if (!validatePhotoAlignmentProblem(problem, error))
        {
            return false;
        }
        if (result.refinedCameras.size() != problem.cameraIds.size() || result.points.size() != problem.trackIds.size())
        {
            setError(error, "solver result camera/track counts do not match the photo-alignment problem");
            return false;
        }

        PhotoAlignmentOutcome converted;
        converted.terminationStatus = result.status;
        converted.terminationReason = result.backendMessage;
        converted.requestedBackend = result.requestedBackend;
        converted.usedBackend = result.usedBackend;
        converted.backendSelectionReason = result.backendSelectionReason;
        converted.solutionUsable = result.solutionUsable;
        converted.backendFallback = result.backendFallback;
        converted.qualityGateRejected = result.qualityGateRejected;
        converted.meanRmsBefore = result.quality.meanRmsBefore;
        converted.meanRmsAfter = result.quality.meanRmsAfter;
        converted.cameras.reserve(problem.cameraIds.size());
        converted.tracks.reserve(problem.trackIds.size());
        for (std::size_t index = 0; index < problem.cameraIds.size(); ++index)
        {
            converted.cameras.push_back({problem.cameraIds[index], result.refinedCameras[index]});
        }
        for (std::size_t index = 0; index < problem.trackIds.size(); ++index)
        {
            converted.tracks.push_back({problem.trackIds[index], result.points[index]});
        }
        if (!validatePhotoAlignmentOutcome(converted, error))
        {
            return false;
        }
        *outcome = std::move(converted);
        return true;
    }

    bool validatePhotoAlignmentOutcome(const PhotoAlignmentOutcome& outcome, std::string* error)
    {
        if (!finite(outcome.meanRmsBefore) || !finite(outcome.meanRmsAfter))
        {
            setError(error, "photo-alignment outcome RMS values must be finite");
            return false;
        }
        std::map<std::string, const PhotoAlignmentCameraState*> cameras;
        std::map<std::string, const PhotoAlignmentTrackState*> tracks;
        if (!makeIndex(outcome.cameras, "camera", &cameras, error) ||
            !makeIndex(outcome.tracks, "track", &tracks, error))
        {
            return false;
        }
        for (const auto& [id, state] : cameras)
        {
            std::string camera_error;
            if (!validateFrameCamera(state->camera, &camera_error))
            {
                setError(error, "photo-alignment camera " + id + " is invalid: " + camera_error);
                return false;
            }
        }
        for (const auto& [id, state] : tracks)
        {
            const auto& point = state->point.point;
            if (!finite(point[0]) || !finite(point[1]) || !finite(point[2]) || !finite(state->point.rmsBefore) ||
                !finite(state->point.rmsAfter))
            {
                setError(error, "photo-alignment track " + id + " contains non-finite state");
                return false;
            }
        }
        return true;
    }

    bool comparePhotoAlignmentOutcomes(const PhotoAlignmentOutcome& reference,
                                       const PhotoAlignmentOutcome& candidate,
                                       const PhotoAlignmentComparisonTolerance& tolerance,
                                       PhotoAlignmentComparison* comparison,
                                       std::string* error)
    {
        if (!comparison)
        {
            setError(error, "photo-alignment comparison output is null");
            return false;
        }
        *comparison = {};
        if (!validTolerance(tolerance, error) || !validatePhotoAlignmentOutcome(reference, error) ||
            !validatePhotoAlignmentOutcome(candidate, error))
        {
            return false;
        }

        std::map<std::string, const PhotoAlignmentCameraState*> reference_cameras;
        std::map<std::string, const PhotoAlignmentCameraState*> candidate_cameras;
        std::map<std::string, const PhotoAlignmentTrackState*> reference_tracks;
        std::map<std::string, const PhotoAlignmentTrackState*> candidate_tracks;
        makeIndex(reference.cameras, "camera", &reference_cameras, nullptr);
        makeIndex(candidate.cameras, "camera", &candidate_cameras, nullptr);
        makeIndex(reference.tracks, "track", &reference_tracks, nullptr);
        makeIndex(candidate.tracks, "track", &candidate_tracks, nullptr);
        if (!sameIds(reference_cameras, candidate_cameras) || !sameIds(reference_tracks, candidate_tracks))
        {
            setError(error, "photo-alignment outcomes must contain the same camera and track ids");
            comparison->message = "camera or track id sets differ";
            return false;
        }

        comparison->comparable = true;
        comparison->terminationStatusMatches = reference.terminationStatus == candidate.terminationStatus;
        comparison->terminationReasonMatches = reference.terminationReason == candidate.terminationReason;
        comparison->requestedBackendMatches = reference.requestedBackend == candidate.requestedBackend;
        comparison->usedBackendMatches = reference.usedBackend == candidate.usedBackend;
        comparison->rmsBeforeDifference = std::abs(reference.meanRmsBefore - candidate.meanRmsBefore);
        comparison->rmsAfterDifference = std::abs(reference.meanRmsAfter - candidate.meanRmsAfter);

        for (const auto& [id, reference_state] : reference_cameras)
        {
            const PhotoAlignmentCameraState& candidate_state = *candidate_cameras.at(id);
            PhotoAlignmentCameraDifference difference;
            difference.id = id;
            difference.centerDistanceMeters =
                distance(reference_state->camera.cameraCenter, candidate_state.camera.cameraCenter);
            difference.rotationAngleDegrees = rotationDifferenceDegrees(reference_state->camera.cameraToWorldRotation,
                                                                        candidate_state.camera.cameraToWorldRotation);
            comparison->maximumCameraCenterDifferenceMeters =
                std::max(comparison->maximumCameraCenterDifferenceMeters, difference.centerDistanceMeters);
            comparison->maximumCameraRotationDifferenceDegrees =
                std::max(comparison->maximumCameraRotationDifferenceDegrees, difference.rotationAngleDegrees);
            comparison->cameraDifferences.push_back(std::move(difference));
        }

        for (const auto& [id, reference_state] : reference_tracks)
        {
            const PhotoAlignmentTrackState& candidate_state = *candidate_tracks.at(id);
            PhotoAlignmentTrackDifference difference;
            difference.id = id;
            difference.referenceValid = reference_state->point.valid;
            difference.candidateValid = candidate_state.point.valid;
            if (difference.referenceValid != difference.candidateValid)
            {
                comparison->validTrackMaskMismatchIds.push_back(id);
            }
            if (difference.referenceValid && difference.candidateValid)
            {
                difference.pointDistanceMeters = distance(reference_state->point.point, candidate_state.point.point);
                comparison->maximumPointDifferenceMeters =
                    std::max(comparison->maximumPointDifferenceMeters, difference.pointDistanceMeters);
            }
            comparison->trackDifferences.push_back(std::move(difference));
        }

        const bool status_ok = !tolerance.requireTerminationStatusMatch || comparison->terminationStatusMatches;
        const bool reason_ok = !tolerance.requireTerminationReasonMatch || comparison->terminationReasonMatches;
        const bool requested_ok = !tolerance.requireRequestedBackendMatch || comparison->requestedBackendMatches;
        const bool used_ok = !tolerance.requireUsedBackendMatch || comparison->usedBackendMatches;
        comparison->equivalent =
            status_ok && reason_ok && requested_ok && used_ok && comparison->rmsBeforeDifference <= tolerance.rms &&
            comparison->rmsAfterDifference <= tolerance.rms &&
            comparison->maximumCameraCenterDifferenceMeters <= tolerance.cameraCenterMeters &&
            comparison->maximumCameraRotationDifferenceDegrees <= tolerance.cameraRotationDegrees &&
            comparison->maximumPointDifferenceMeters <= tolerance.pointMeters &&
            comparison->validTrackMaskMismatchIds.empty();
        comparison->message =
            comparison->equivalent ? "photo-alignment outcomes are equivalent" : "photo-alignment outcomes differ";
        return true;
    }

} // namespace plabundle
