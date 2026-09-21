#pragma once

#include <array>
#include <optional>
#include <string>

namespace plabundle
{

    struct ImageSize
    {
        int samples = 0;
        int lines = 0;

        bool valid() const noexcept
        {
            return samples > 0 && lines > 0;
        }
    };

    struct BrownConradyDistortion
    {
        double k1 = 0.0;
        double k2 = 0.0;
        double k3 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;
    };

    struct FrameCamera
    {
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
        double focalXPixels = 0.0;
        double focalYPixels = 0.0;
        double principalXPixel = 0.0;
        double principalYPixel = 0.0;
        double pixelPitchMillimeters = 1.0;
        BrownConradyDistortion distortion;
        int uAxisSign = 1;
        int vAxisSign = 1;
        bool depthAxisFlipped = false;
        std::optional<ImageSize> imageSize;
    };

    struct Projection
    {
        std::array<double, 2> pixel{{0.0, 0.0}};
        double positiveDepth = 0.0;
    };

    bool validateFrameCamera(const FrameCamera& camera, std::string* error = nullptr) noexcept;
    std::array<double, 3> worldToCameraPoint(const FrameCamera& camera, const std::array<double, 3>& world) noexcept;
    double positiveDepth(const FrameCamera& camera, const std::array<double, 3>& world) noexcept;
    bool
    projectWorldPoint(const FrameCamera& camera, const std::array<double, 3>& world, Projection* projection) noexcept;
    bool applyPoseDelta(FrameCamera* camera, const std::array<double, 6>& delta, std::string* error = nullptr) noexcept;

} // namespace plabundle
