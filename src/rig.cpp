#include <plabundle/rig.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace plabundle
{
    namespace
    {
        using Key = std::pair<int, int>;

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

        template <std::size_t Size> bool finiteArray(const std::array<double, Size>& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }

        bool validRotation(const std::array<double, 9>& rotation) noexcept
        {
            if (!finiteArray(rotation))
            {
                return false;
            }
            for (int first = 0; first < 3; ++first)
            {
                for (int second = first; second < 3; ++second)
                {
                    double dot = 0.0;
                    for (int row = 0; row < 3; ++row)
                    {
                        dot += rotation[static_cast<std::size_t>(row * 3 + first)] *
                               rotation[static_cast<std::size_t>(row * 3 + second)];
                    }
                    const double expected = first == second ? 1.0 : 0.0;
                    if (std::abs(dot - expected) > 1.0e-6)
                    {
                        return false;
                    }
                }
            }
            const double determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                                       rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                                       rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
            return std::abs(determinant - 1.0) <= 1.0e-6;
        }

        std::array<double, 9> multiply(const std::array<double, 9>& left, const std::array<double, 9>& right) noexcept
        {
            std::array<double, 9> product{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int inner = 0; inner < 3; ++inner)
                    {
                        product[static_cast<std::size_t>(row * 3 + column)] +=
                            left[static_cast<std::size_t>(row * 3 + inner)] *
                            right[static_cast<std::size_t>(inner * 3 + column)];
                    }
                }
            }
            return product;
        }

        std::array<double, 3> transform(const std::array<double, 9>& rotation,
                                        const std::array<double, 3>& point) noexcept
        {
            std::array<double, 3> result{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    result[static_cast<std::size_t>(row)] +=
                        rotation[static_cast<std::size_t>(row * 3 + column)] * point[static_cast<std::size_t>(column)];
                }
            }
            return result;
        }

    } // namespace

    bool validateRigTopology(const RigTopology& topology, std::size_t cameraCount, std::string* error)
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
        if (topology.empty())
        {
            return true;
        }
        if (topology.captures.empty() || topology.sensors.empty() || topology.cameraBindings.size() != cameraCount)
        {
            setError(error, "rig topology requires captures, sensors, and exactly one binding per camera");
            return false;
        }

        std::set<Key> capture_keys;
        for (const RigCapture& capture : topology.captures)
        {
            if (capture.rigId < 0 || capture.captureId < 0 || !validRotation(capture.rigToWorldRotation) ||
                !finiteArray(capture.rigCenterInWorld) ||
                !capture_keys.emplace(capture.rigId, capture.captureId).second)
            {
                setError(error, "rig captures must have unique non-negative ids and finite rigid poses");
                return false;
            }
        }

        std::set<Key> sensor_keys;
        std::set<int> rigs_with_fixed_sensor;
        for (const RigSensor& sensor : topology.sensors)
        {
            if (sensor.rigId < 0 || sensor.sensorId < 0 || !validRotation(sensor.cameraToRigRotation) ||
                !finiteArray(sensor.cameraCenterInRig) || !sensor_keys.emplace(sensor.rigId, sensor.sensorId).second)
            {
                setError(error, "rig sensors must have unique non-negative ids and finite rigid extrinsics");
                return false;
            }
            if (sensor.fixedExtrinsic)
            {
                rigs_with_fixed_sensor.insert(sensor.rigId);
            }
        }
        for (const RigCapture& capture : topology.captures)
        {
            if (rigs_with_fixed_sensor.count(capture.rigId) == 0)
            {
                setError(error, "each rig requires at least one fixed sensor extrinsic to define its body frame");
                return false;
            }
        }

        std::set<int> camera_indices;
        for (const RigCameraBinding& binding : topology.cameraBindings)
        {
            if (binding.cameraIndex < 0 || static_cast<std::size_t>(binding.cameraIndex) >= cameraCount ||
                binding.rigId < 0 || binding.captureId < 0 || binding.sensorId < 0 ||
                !camera_indices.insert(binding.cameraIndex).second ||
                capture_keys.count({binding.rigId, binding.captureId}) == 0 ||
                sensor_keys.count({binding.rigId, binding.sensorId}) == 0)
            {
                setError(error, "rig camera bindings must be unique and reference an existing capture and sensor");
                return false;
            }
        }
        return camera_indices.size() == cameraCount;
    }

    bool composeRigCameras(const std::vector<FrameCamera>& cameraModels,
                           const RigTopology& topology,
                           std::vector<FrameCamera>* cameras,
                           std::string* error)
    {
        if (!cameras)
        {
            setError(error, "rig camera output is null");
            return false;
        }
        if (!validateRigTopology(topology, cameraModels.size(), error))
        {
            return false;
        }
        if (topology.empty())
        {
            *cameras = cameraModels;
            return true;
        }

        std::map<Key, const RigCapture*> captures;
        std::map<Key, const RigSensor*> sensors;
        for (const RigCapture& capture : topology.captures)
        {
            captures[{capture.rigId, capture.captureId}] = &capture;
        }
        for (const RigSensor& sensor : topology.sensors)
        {
            sensors[{sensor.rigId, sensor.sensorId}] = &sensor;
        }

        std::vector<FrameCamera> result = cameraModels;
        for (const RigCameraBinding& binding : topology.cameraBindings)
        {
            const RigCapture& capture = *captures.at({binding.rigId, binding.captureId});
            const RigSensor& sensor = *sensors.at({binding.rigId, binding.sensorId});
            FrameCamera& camera = result[static_cast<std::size_t>(binding.cameraIndex)];
            camera.cameraToWorldRotation = multiply(capture.rigToWorldRotation, sensor.cameraToRigRotation);
            const auto offset = transform(capture.rigToWorldRotation, sensor.cameraCenterInRig);
            for (int axis = 0; axis < 3; ++axis)
            {
                camera.cameraCenter[static_cast<std::size_t>(axis)] =
                    capture.rigCenterInWorld[static_cast<std::size_t>(axis)] + offset[static_cast<std::size_t>(axis)];
            }
            if (!validateFrameCamera(camera, error))
            {
                setError(error, "composed rig camera " + std::to_string(binding.cameraIndex) + " is invalid");
                return false;
            }
        }
        *cameras = std::move(result);
        return true;
    }

} // namespace plabundle
