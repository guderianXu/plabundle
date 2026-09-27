#pragma once

#include <plabundle/camera.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace plabundle::internal
{

    class CameraState final
    {
    public:
        struct Intrinsics
        {
            double focalX = 0.0;
            double focalY = 0.0;
            double principalX = 0.0;
            double principalY = 0.0;
            double pixelPitch = 1.0;
            int uAxisSign = 1;
            int vAxisSign = 1;
        };

        struct Distortion
        {
            double radialK1 = 0.0;
            double radialK2 = 0.0;
            double radialK3 = 0.0;
            double tangentialP1 = 0.0;
            double tangentialP2 = 0.0;
            double radialK4 = 0.0;
            double tangentialP3 = 0.0;
            double tangentialP4 = 0.0;
        };

        CameraState() = default;
        explicit CameraState(FrameCamera camera);

        bool validateNumericalState(std::string* error = nullptr) const noexcept;
        bool isValid() const noexcept;
        std::optional<ImageSize> imageSize() const noexcept;
        Intrinsics intrinsics() const noexcept;
        Distortion distortion() const noexcept;
        std::array<double, 9> cameraToWorldRotation() const noexcept;
        std::array<double, 3> cameraCenter() const noexcept;
        double focalX() const noexcept;
        double focalY() const noexcept;
        double principalX() const noexcept;
        double principalY() const noexcept;
        int uAxisSign() const noexcept;
        int vAxisSign() const noexcept;
        bool depthAxisFlipped() const noexcept;
        FrameProjectionModel projectionModel() const noexcept;
        CameraParameterBlock parameterBlock() const noexcept;
        CameraParameterMask supportedParameters() const noexcept;

        bool projectWorldPoint(const double world[3], double pixel[2]) const noexcept;
        bool projectWorldPointWithDepth(const double world[3], double pixel[2], double& depth) const noexcept;
        bool projectWorldPointWithDepthAtLine(const double world[3],
                                              double linePixels,
                                              double pixel[2],
                                              double& depth) const noexcept;
        void worldToCamera(const double world[3], double cameraPoint[3]) const noexcept;
        void worldToCameraAtLine(const double world[3], double linePixels, double cameraPoint[3]) const noexcept;

        void setPose(const std::array<double, 9>& rotation, const std::array<double, 3>& center) noexcept;
        void setCameraCenter(const std::array<double, 3>& center) noexcept;
        void setIntrinsics(double focalX, double focalY, double principalX, double principalY) noexcept;
        void setImageSize(ImageSize imageSize) noexcept;
        void setAxisDirections(int uDirection, int vDirection) noexcept;
        void setDistortion(const Distortion& distortion) noexcept;
        void setDistortion(double k1, double k2, double k3, double p1, double p2) noexcept;
        bool setParameterBlock(const CameraParameterBlock& parameters,
                               const CameraParameterMask& mask,
                               std::string* error = nullptr) noexcept;
        void applyDeltaPose(const double delta[6]) noexcept;
        CameraState normalizedForPositiveDepth() const noexcept;

        const FrameCamera& frameCamera() const noexcept;

    private:
        FrameCamera _camera;
    };

    std::vector<CameraState> makeCameraStates(const std::vector<FrameCamera>& cameras);
    std::vector<FrameCamera> makeFrameCameras(const std::vector<CameraState>& cameras);

} // namespace plabundle::internal
