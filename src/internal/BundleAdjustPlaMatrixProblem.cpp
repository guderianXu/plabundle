#include "BundleAdjustPlaMatrixProblem.h"

#include "BundleAdjustValidation.h"

#include <plamatrix/dense/matrix.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace plabundle::internal::plamatrix_ba
{
    namespace
    {

        bool isCameraFixed(int camera_index, const BAOptions& options)
        {
            return std::find(options.fixedCameraIndices.begin(), options.fixedCameraIndices.end(), camera_index) !=
                   options.fixedCameraIndices.end();
        }

        bool isTrackFixed(int track_index, const BAOptions& options)
        {
            return std::binary_search(options.fixedTrackIndices.begin(), options.fixedTrackIndices.end(), track_index);
        }

    } // namespace

    ActiveProblem prepareActiveProblem(const std::vector<CameraState>& cameras,
                                       const std::vector<BATrack>& tracks,
                                       const BAOptions& options)
    {
        ActiveProblem problem;
        problem.activeTrack.assign(tracks.size(), 0);
        problem.activeLaserRange.assign(options.laserRangeConstraints.size(), 0);
        problem.cameraBlock.assign(cameras.size(), -1);
        problem.rigCaptureBlockByCamera.assign(cameras.size(), -1);
        problem.rigSensorBlockByCamera.assign(cameras.size(), -1);
        problem.rigCaptureIndexByCamera.assign(cameras.size(), -1);
        problem.rigSensorIndexByCamera.assign(cameras.size(), -1);
        problem.intrinsicBlockByCamera.assign(cameras.size(), -1);
        problem.calibrationGroupByCamera.assign(cameras.size(), 0);
        problem.trackPrimaryBlock.assign(tracks.size(), -1);
        problem.trackBlock.assign(tracks.size(), -1);
        problem.laserBlock.assign(options.laserRangeConstraints.size(), -1);
        std::vector<char> camera_has_residual(cameras.size(), 0);
        if (!options.rig.empty())
        {
            for (const placamera::RigCameraBinding& binding : options.rig.cameraBindings)
            {
                const auto capture =
                    std::find_if(options.rig.captures.begin(),
                                 options.rig.captures.end(),
                                 [&](const placamera::RigCapture& value)
                                 { return value.rigId == binding.rigId && value.captureId == binding.captureId; });
                const auto sensor =
                    std::find_if(options.rig.sensors.begin(),
                                 options.rig.sensors.end(),
                                 [&](const placamera::RigSensor& value)
                                 { return value.rigId == binding.rigId && value.sensorId == binding.sensorId; });
                const std::size_t camera_index = static_cast<std::size_t>(binding.cameraIndex);
                problem.rigCaptureIndexByCamera[camera_index] =
                    static_cast<int>(std::distance(options.rig.captures.begin(), capture));
                problem.rigSensorIndexByCamera[camera_index] =
                    static_cast<int>(std::distance(options.rig.sensors.begin(), sensor));
            }
        }

        for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
        {
            const BATrack& track = tracks[track_index];
            if (!plamatrix::Vector3d(track.initialPoint[0], track.initialPoint[1], track.initialPoint[2]).allFinite())
            {
                continue;
            }
            int first_camera = -1;
            bool second_camera = false;
            bool positive_depth = true;
            int observation_count = 0;
            double squared_residual = 0.0;
            for (const BAObservation& observation : track.observations)
            {
                if (!observationIsUsable(observation, cameras.size()))
                {
                    continue;
                }
                first_camera = first_camera < 0 ? observation.cameraIndex : first_camera;
                second_camera = second_camera || observation.cameraIndex != first_camera;
                double pixel[2] = {0.0, 0.0};
                double depth = 0.0;
                const double world[3] = {track.initialPoint[0], track.initialPoint[1], track.initialPoint[2]};
                if (!cameras[static_cast<std::size_t>(observation.cameraIndex)].projectWorldPointWithDepthAtLine(
                        world, observation.v, pixel, depth))
                {
                    positive_depth = false;
                    break;
                }
                const double du = pixel[0] - observation.u;
                const double dv = pixel[1] - observation.v;
                squared_residual += sanitizedObservationWeight(observation) * (du * du + dv * dv);
                ++observation_count;
            }
            const double initial_rms = observation_count > 0
                                           ? std::sqrt(squared_residual / static_cast<double>(2 * observation_count))
                                           : std::numeric_limits<double>::infinity();
            if (observation_count < 2 || !second_camera || !positive_depth)
            {
                continue;
            }
            if (options.maxInitialTrackRms > 0.0 && initial_rms > options.maxInitialTrackRms)
            {
                ++problem.rejectedInitialTracks;
                continue;
            }

            problem.activeTrack[track_index] = 1;
            ++problem.activeTrackCount;
            problem.observationCount += observation_count;
            for (const BAObservation& observation : track.observations)
            {
                if (observationIsUsable(observation, cameras.size()))
                {
                    camera_has_residual[static_cast<std::size_t>(observation.cameraIndex)] = 1;
                }
            }
        }

        if (options.enableLaserRangeConstraints)
        {
            for (std::size_t shot_index = 0; shot_index < options.laserRangeConstraints.size(); ++shot_index)
            {
                const auto& constraint = options.laserRangeConstraints[shot_index];
                problem.activeLaserRange[shot_index] = 1;
                ++problem.activeLaserRangeCount;
                camera_has_residual[static_cast<std::size_t>(constraint.cameraIndex)] = 1;
                for (const auto& observation : constraint.measuredImageObservations)
                {
                    if (observationIsUsable(observation, cameras.size()))
                    {
                        camera_has_residual[static_cast<std::size_t>(observation.cameraIndex)] = 1;
                    }
                }
            }
        }

        if (options.refineCameraPose && options.rig.empty())
        {
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
            {
                if (camera_has_residual[camera_index] && !isCameraFixed(static_cast<int>(camera_index), options))
                {
                    problem.cameraBlock[camera_index] = problem.cameraBlockCount++;
                }
            }
        }
        else if (options.refineCameraPose)
        {
            std::vector<int> capture_blocks(options.rig.captures.size(), -1);
            std::vector<int> sensor_blocks(options.rig.sensors.size(), -1);
            for (std::size_t capture_index = 0; capture_index < options.rig.captures.size(); ++capture_index)
            {
                bool has_residual = false;
                bool fixed = options.rig.captures[capture_index].fixedPose;
                for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
                {
                    if (problem.rigCaptureIndexByCamera[camera_index] == static_cast<int>(capture_index))
                    {
                        has_residual = has_residual || camera_has_residual[camera_index];
                        fixed = fixed || isCameraFixed(static_cast<int>(camera_index), options) ||
                                options.referenceGaugeScaleCameraIndex == static_cast<int>(camera_index);
                    }
                }
                if (has_residual && !fixed)
                {
                    capture_blocks[capture_index] = problem.rigCaptureBlockCount++;
                }
            }
            for (std::size_t sensor_index = 0; sensor_index < options.rig.sensors.size(); ++sensor_index)
            {
                bool has_residual = false;
                for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
                {
                    has_residual = has_residual ||
                                   (problem.rigSensorIndexByCamera[camera_index] == static_cast<int>(sensor_index) &&
                                    camera_has_residual[camera_index]);
                }
                if (has_residual && !options.rig.sensors[sensor_index].fixedExtrinsic)
                {
                    sensor_blocks[sensor_index] = problem.rigCaptureBlockCount + problem.rigSensorBlockCount++;
                }
            }
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
            {
                problem.rigCaptureBlockByCamera[camera_index] =
                    capture_blocks[static_cast<std::size_t>(problem.rigCaptureIndexByCamera[camera_index])];
                problem.rigSensorBlockByCamera[camera_index] =
                    sensor_blocks[static_cast<std::size_t>(problem.rigSensorIndexByCamera[camera_index])];
            }
            problem.cameraBlockCount = problem.rigCaptureBlockCount + problem.rigSensorBlockCount;
        }
        problem.primaryBlockCount = problem.cameraBlockCount;

        if (hasSharedIntrinsics(options))
        {
            if (!options.cameraCalibrationGroupIds.empty())
            {
                problem.calibrationGroupByCamera = options.cameraCalibrationGroupIds;
            }
            std::vector<int> group_ids;
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
            {
                const int group_id = problem.calibrationGroupByCamera[camera_index];
                auto found = std::find(group_ids.begin(), group_ids.end(), group_id);
                int group_index = 0;
                if (found == group_ids.end())
                {
                    group_index = static_cast<int>(group_ids.size());
                    group_ids.push_back(group_id);
                }
                else
                {
                    group_index = static_cast<int>(std::distance(group_ids.begin(), found));
                }
                problem.calibrationGroupByCamera[camera_index] = group_index;
            }
            problem.intrinsicBlockCount = static_cast<int>(group_ids.size());
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
            {
                problem.intrinsicBlockByCamera[camera_index] =
                    problem.primaryBlockCount + problem.calibrationGroupByCamera[camera_index];
            }
            problem.primaryBlockCount += problem.intrinsicBlockCount;
        }

        std::vector<char> promote_track(tracks.size(), 0);
        if (options.enableScaleBarConstraints)
        {
            for (const auto& constraint : options.scaleBarConstraints)
            {
                for (const int track_index : {constraint.trackIndexA, constraint.trackIndexB})
                {
                    if (track_index >= 0 && track_index < static_cast<int>(tracks.size()) &&
                        problem.activeTrack[static_cast<std::size_t>(track_index)] &&
                        !isTrackFixed(track_index, options))
                    {
                        promote_track[static_cast<std::size_t>(track_index)] = 1;
                    }
                }
            }
        }
        for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
        {
            if (!problem.activeTrack[track_index] || isTrackFixed(static_cast<int>(track_index), options))
            {
                continue;
            }
            if (promote_track[track_index])
            {
                problem.trackPrimaryBlock[track_index] = problem.primaryBlockCount++;
                ++problem.promotedTrackBlockCount;
            }
            else
            {
                problem.trackBlock[track_index] = problem.trackBlockCount++;
            }
        }
        if (options.enableLaserRangeConstraints)
        {
            for (std::size_t shot_index = 0; shot_index < options.laserRangeConstraints.size(); ++shot_index)
            {
                if (options.laserRangeConstraints[shot_index].pointMode != BALaserPointMode::Fixed)
                {
                    problem.laserBlock[shot_index] = problem.trackBlockCount + problem.laserBlockCount++;
                }
            }
        }
        return problem;
    }

} // namespace plabundle::internal::plamatrix_ba
