#pragma once

#include <plabundle/camera.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace plabundle
{

    struct RigCapture
    {
        int rigId = -1;
        int captureId = -1;
        std::array<double, 9> rigToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> rigCenterInWorld{{0.0, 0.0, 0.0}};
        bool fixedPose = false;
    };

    struct RigSensor
    {
        int rigId = -1;
        int sensorId = -1;
        std::array<double, 9> cameraToRigRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenterInRig{{0.0, 0.0, 0.0}};
        bool fixedExtrinsic = true;
    };

    struct RigCameraBinding
    {
        int cameraIndex = -1;
        int rigId = -1;
        int captureId = -1;
        int sensorId = -1;
    };

    struct RigTopology
    {
        std::vector<RigCapture> captures;
        std::vector<RigSensor> sensors;
        std::vector<RigCameraBinding> cameraBindings;

        bool empty() const noexcept
        {
            return captures.empty() && sensors.empty() && cameraBindings.empty();
        }
    };

    bool validateRigTopology(const RigTopology& topology, std::size_t cameraCount, std::string* error = nullptr);

    bool composeRigCameras(const std::vector<FrameCamera>& cameraModels,
                           const RigTopology& topology,
                           std::vector<FrameCamera>* cameras,
                           std::string* error = nullptr);

} // namespace plabundle
