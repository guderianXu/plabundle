#include <plabundle/problem.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>

namespace plabundle
{
    namespace
    {
        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        template <std::size_t Size> bool finiteArray(const std::array<double, Size>& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return finite(value); });
        }

        void setError(std::string* error, const std::string& message) noexcept
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

        bool validObservation(const Observation& observation, std::size_t camera_count) noexcept
        {
            return observation.cameraIndex >= 0 && static_cast<std::size_t>(observation.cameraIndex) < camera_count &&
                   finite(observation.u) && finite(observation.v) && finite(observation.weight) &&
                   observation.weight > 0.0 && finite(observation.measurementScale) &&
                   observation.measurementScale > 0.0;
        }

        bool validIndexList(const std::vector<int>& indices, std::size_t size)
        {
            std::set<int> unique;
            for (int index : indices)
            {
                if (index < 0 || static_cast<std::size_t>(index) >= size || !unique.insert(index).second)
                {
                    return false;
                }
            }
            return true;
        }

        bool validPosePrior(const CameraPosePrior& prior) noexcept
        {
            FrameCamera camera;
            camera.cameraToWorldRotation = prior.cameraToWorldRotation;
            camera.cameraCenter = prior.cameraCenter;
            camera.focalXPixels = 1.0;
            camera.focalYPixels = 1.0;
            return validateFrameCamera(camera) && finite(prior.positionSigmaMeters) &&
                   prior.positionSigmaMeters > 0.0 && finite(prior.rotationSigmaDegrees) &&
                   prior.rotationSigmaDegrees > 0.0;
        }

        bool usableSqrtInformation(const std::array<double, 9>& matrix) noexcept
        {
            double maximum = 0.0;
            for (double value : matrix)
            {
                if (!finite(value))
                {
                    return false;
                }
                maximum = std::max(maximum, std::abs(value));
            }
            if (!(maximum > 0.0))
            {
                return false;
            }
            const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                                       matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                                       matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
            const double threshold = 1.0e-12 * maximum * maximum * maximum;
            return finite(determinant) && std::abs(determinant) > threshold;
        }

        double squaredDistance(const std::array<double, 3>& left, const std::array<double, 3>& right) noexcept
        {
            const double dx = left[0] - right[0];
            const double dy = left[1] - right[1];
            const double dz = left[2] - right[2];
            return dx * dx + dy * dy + dz * dz;
        }

    } // namespace

    ProblemStats summarizeProblem(const Problem& problem)
    {
        ProblemStats stats;
        stats.cameraCount = static_cast<int>(problem.cameras.size());
        for (const Track& track : problem.tracks)
        {
            if (!finiteArray(track.initialPoint))
            {
                continue;
            }
            std::set<int> cameras;
            int observations = 0;
            for (const Observation& observation : track.observations)
            {
                if (validObservation(observation, problem.cameras.size()))
                {
                    cameras.insert(observation.cameraIndex);
                    ++observations;
                }
            }
            if (cameras.size() >= 2)
            {
                ++stats.trackCount;
                stats.observationCount += observations;
            }
        }
        return stats;
    }

    bool validateProblem(const Problem& problem, std::string* error)
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
        if (problem.cameras.size() < 2)
        {
            setError(error, "problem requires at least two cameras");
            return false;
        }
        if (problem.tracks.empty())
        {
            setError(error, "problem requires at least one track");
            return false;
        }
        for (std::size_t index = 0; index < problem.cameras.size(); ++index)
        {
            std::string camera_error;
            if (!validateFrameCamera(problem.cameras[index], &camera_error))
            {
                setError(error, "camera " + std::to_string(index) + " is invalid: " + camera_error);
                return false;
            }
        }
        if (!validIndexList(problem.fixedCameraIndices, problem.cameras.size()) ||
            !validIndexList(problem.fixedTrackIndices, problem.tracks.size()))
        {
            setError(error, "fixed camera/track indices must be unique and in range");
            return false;
        }
        if (!problem.cameraCalibrationGroupIds.empty() &&
            problem.cameraCalibrationGroupIds.size() != problem.cameras.size())
        {
            setError(error, "camera calibration groups must be empty or aligned with cameras");
            return false;
        }
        if (std::any_of(problem.cameraCalibrationGroupIds.begin(),
                        problem.cameraCalibrationGroupIds.end(),
                        [](int group) { return group < 0; }))
        {
            setError(error, "camera calibration group ids must be non-negative");
            return false;
        }
        if (!problem.sharedIntrinsicReferenceCameras.empty() &&
            problem.sharedIntrinsicReferenceCameras.size() != problem.cameras.size())
        {
            setError(error, "shared intrinsic reference cameras must be empty or aligned with cameras");
            return false;
        }
        for (const FrameCamera& camera : problem.sharedIntrinsicReferenceCameras)
        {
            if (!validateFrameCamera(camera))
            {
                setError(error, "shared intrinsic reference camera is invalid");
                return false;
            }
        }

        for (std::size_t track_index = 0; track_index < problem.tracks.size(); ++track_index)
        {
            const Track& track = problem.tracks[track_index];
            if (!finiteArray(track.initialPoint))
            {
                setError(error, "track " + std::to_string(track_index) + " has a non-finite initial point");
                return false;
            }
            std::set<int> observing_cameras;
            for (const Observation& observation : track.observations)
            {
                if (!validObservation(observation, problem.cameras.size()))
                {
                    setError(error, "track " + std::to_string(track_index) + " has an invalid observation");
                    return false;
                }
                observing_cameras.insert(observation.cameraIndex);
            }
            if (observing_cameras.size() < 2)
            {
                setError(error,
                         "track " + std::to_string(track_index) + " must be observed by at least two distinct cameras");
                return false;
            }
            for (const LaserPlaneConstraint& constraint : track.laserPlaneConstraints)
            {
                const double normal_norm = std::hypot(constraint.normal[0], constraint.normal[1], constraint.normal[2]);
                if (!finiteArray(constraint.point) || !finiteArray(constraint.normal) || !finite(normal_norm) ||
                    normal_norm <= 0.0 || !finite(constraint.weight) || constraint.weight <= 0.0 ||
                    !finite(constraint.initialSignedDistance))
                {
                    setError(error, "track contains an invalid laser-plane constraint");
                    return false;
                }
            }
            for (const ControlPointConstraint& constraint : track.controlPointConstraints)
            {
                if (!finiteArray(constraint.point) || !finite(constraint.sigmaMeters) ||
                    constraint.sigmaMeters <= 0.0 || !finite(constraint.weight) || constraint.weight <= 0.0)
                {
                    setError(error, "track contains an invalid control-point constraint");
                    return false;
                }
            }
        }

        for (const LaserRangeConstraint& constraint : problem.laserRangeConstraints)
        {
            const int point_mode = static_cast<int>(constraint.pointMode);
            if (constraint.cameraIndex < 0 ||
                static_cast<std::size_t>(constraint.cameraIndex) >= problem.cameras.size() ||
                point_mode <= static_cast<int>(LaserPointMode::Unspecified) ||
                point_mode > static_cast<int>(LaserPointMode::Free) || !finiteArray(constraint.initialPoint) ||
                !finiteArray(constraint.leverArmCameraMeters) || !finiteArray(constraint.pointPrior) ||
                !finiteArray(constraint.pointPriorSqrtInformation) || !finite(constraint.observedRangeMeters) ||
                constraint.observedRangeMeters <= 0.0 || !finite(constraint.sigmaRangeMeters) ||
                constraint.sigmaRangeMeters <= 0.0 || !finite(constraint.weight) || constraint.weight <= 0.0 ||
                !finite(constraint.ephemerisTimeSeconds))
            {
                setError(error, "problem contains an invalid laser-range constraint");
                return false;
            }
            const FrameCamera& source_camera = problem.cameras[static_cast<std::size_t>(constraint.cameraIndex)];
            std::array<double, 3> emitter = source_camera.cameraCenter;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    emitter[static_cast<std::size_t>(row)] +=
                        source_camera.cameraToWorldRotation[static_cast<std::size_t>(row * 3 + column)] *
                        constraint.leverArmCameraMeters[static_cast<std::size_t>(column)];
                }
            }
            const double initial_range_squared = squaredDistance(constraint.initialPoint, emitter);
            if (!finite(initial_range_squared) || initial_range_squared <= 1.0e-18)
            {
                setError(error, "laser-range initial point must differ from the lever-arm-adjusted emitter");
                return false;
            }
            std::set<int> measured_cameras;
            for (const Observation& observation : constraint.measuredImageObservations)
            {
                if (!validObservation(observation, problem.cameras.size()))
                {
                    setError(error, "laser-range constraint contains an invalid image observation");
                    return false;
                }
                measured_cameras.insert(observation.cameraIndex);
                Projection projection;
                if (!projectWorldPoint(problem.cameras[static_cast<std::size_t>(observation.cameraIndex)],
                                       constraint.initialPoint,
                                       &projection))
                {
                    setError(error, "laser-range initial point must project in front of each measured camera");
                    return false;
                }
            }
            if (constraint.pointMode == LaserPointMode::Constrained &&
                (!finiteArray(constraint.pointPrior) || !usableSqrtInformation(constraint.pointPriorSqrtInformation)))
            {
                setError(error, "constrained laser point requires a finite prior and full-rank information matrix");
                return false;
            }
            if (constraint.pointMode == LaserPointMode::Free)
            {
                bool has_baseline = false;
                for (auto left = measured_cameras.begin(); left != measured_cameras.end() && !has_baseline; ++left)
                {
                    auto right = std::next(left);
                    for (; right != measured_cameras.end(); ++right)
                    {
                        if (squaredDistance(problem.cameras[static_cast<std::size_t>(*left)].cameraCenter,
                                            problem.cameras[static_cast<std::size_t>(*right)].cameraCenter) > 1.0e-16)
                        {
                            has_baseline = true;
                            break;
                        }
                    }
                }
                if (constraint.measuredImageObservations.size() < 2 || measured_cameras.size() < 2 || !has_baseline)
                {
                    setError(error, "free laser point requires measured observations from two baseline cameras");
                    return false;
                }
            }
        }
        for (const ScaleBarConstraint& constraint : problem.scaleBarConstraints)
        {
            if (constraint.trackIndexA < 0 || constraint.trackIndexB < 0 ||
                constraint.trackIndexA == constraint.trackIndexB ||
                static_cast<std::size_t>(constraint.trackIndexA) >= problem.tracks.size() ||
                static_cast<std::size_t>(constraint.trackIndexB) >= problem.tracks.size() ||
                !finite(constraint.measuredDistanceMeters) || constraint.measuredDistanceMeters <= 0.0 ||
                !finite(constraint.sigmaMeters) || constraint.sigmaMeters <= 0.0 || !finite(constraint.weight) ||
                constraint.weight <= 0.0)
            {
                setError(error, "problem contains an invalid scale-bar constraint");
                return false;
            }
        }
        if (!problem.cameraPosePriors.empty() && problem.cameraPosePriors.size() != problem.cameras.size())
        {
            setError(error, "camera pose priors must be empty or aligned with cameras");
            return false;
        }
        for (const std::optional<CameraPosePrior>& prior : problem.cameraPosePriors)
        {
            if (prior && !validPosePrior(*prior))
            {
                setError(error, "problem contains an invalid camera pose prior");
                return false;
            }
        }
        if (problem.cameraPlaneConstraint)
        {
            const CameraPlaneConstraint& constraint = *problem.cameraPlaneConstraint;
            const double normal_norm_squared = constraint.normal[0] * constraint.normal[0] +
                                               constraint.normal[1] * constraint.normal[1] +
                                               constraint.normal[2] * constraint.normal[2];
            if (!finiteArray(constraint.point) || !finiteArray(constraint.normal) || !finite(normal_norm_squared) ||
                std::abs(normal_norm_squared - 1.0) > 1.0e-6 || !finite(constraint.sigmaMeters) ||
                constraint.sigmaMeters <= 0.0 || !finite(constraint.weight) || constraint.weight <= 0.0 ||
                (!constraint.referenceSignedDistances.empty() &&
                 constraint.referenceSignedDistances.size() != problem.cameras.size()) ||
                !std::all_of(constraint.referenceSignedDistances.begin(),
                             constraint.referenceSignedDistances.end(),
                             [](double value) { return finite(value); }))
            {
                setError(error, "problem contains an invalid camera-plane constraint");
                return false;
            }
        }
        const Gauge& gauge = problem.gauge;
        const bool has_anchor = gauge.referenceAnchorCameraIndex >= 0;
        const bool has_scale = gauge.referenceScaleCameraIndex >= 0;
        const bool has_requested_gauge = has_anchor || has_scale || gauge.referenceBaseline > 0.0;
        const int gauge_policy = static_cast<int>(gauge.policy);
        if (gauge_policy < static_cast<int>(GaugePolicy::AutoAnchor) ||
            gauge_policy > static_cast<int>(GaugePolicy::CallerManaged) || !finite(gauge.referenceBaseline) ||
            gauge.referenceBaseline < 0.0 || gauge.referenceAnchorCameraIndex < -1 ||
            gauge.referenceScaleCameraIndex < -1 ||
            (has_requested_gauge &&
             (!has_anchor || !has_scale ||
              static_cast<std::size_t>(gauge.referenceAnchorCameraIndex) >= problem.cameras.size() ||
              static_cast<std::size_t>(gauge.referenceScaleCameraIndex) >= problem.cameras.size() ||
              gauge.referenceAnchorCameraIndex == gauge.referenceScaleCameraIndex || gauge.referenceBaseline <= 0.0)))
        {
            setError(error, "reference gauge indices and optional scale baseline are invalid");
            return false;
        }
        return true;
    }

} // namespace plabundle
