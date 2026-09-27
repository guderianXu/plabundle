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
        double k4 = 0.0;
        double p3 = 0.0;
        double p4 = 0.0;
    };

    enum class BrownTangentialConvention
    {
        OpenCv,
        Metashape,
    };

    struct MetashapeFrameCalibration
    {
        double f = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        double b1 = 0.0;
        double b2 = 0.0;
        double k1 = 0.0;
        double k2 = 0.0;
        double k3 = 0.0;
        double k4 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;
        double p3 = 0.0;
        double p4 = 0.0;
    };

    enum class FrameProjectionModel
    {
        BrownConrady,
        Fisheye,
        Equidistant,
        Equisolid,
        Spherical,
        Cylindrical,
        RollingShutter,
    };

    struct RollingShutterMotion
    {
        double referenceLinePixels = 0.0;
        double secondsPerLine = 0.0;
        std::array<double, 3> linearVelocityWorldMetersPerSecond{};
        std::array<double, 3> angularVelocityCameraRadiansPerSecond{};
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
        FrameProjectionModel projectionModel = FrameProjectionModel::BrownConrady;
        std::array<double, 5> modelCoefficients{};
        RollingShutterMotion rollingShutter;
        double skewPixels = 0.0;
        BrownTangentialConvention brownTangentialConvention = BrownTangentialConvention::OpenCv;
    };

    // The first nine entries retain the historical layout. Metashape b1 is
    // represented losslessly by fx/fy: f=fy and b1=fx-fy.
    inline constexpr std::size_t kCameraParameterCount = 13;
    using CameraParameterBlock = std::array<double, kCameraParameterCount>;
    using CameraParameterMask = std::array<bool, kCameraParameterCount>;

    struct Projection
    {
        std::array<double, 2> pixel{{0.0, 0.0}};
        double positiveDepth = 0.0;
    };

    struct ProjectionLinearization
    {
        Projection projection;
        std::array<double, 6> cameraPointJacobian{};
        std::array<double, 2 * kCameraParameterCount> parameterJacobian{};
    };

    bool validateFrameCamera(const FrameCamera& camera, std::string* error = nullptr) noexcept;
    CameraParameterBlock cameraParameterBlock(const FrameCamera& camera) noexcept;
    CameraParameterMask supportedCameraParameters(FrameProjectionModel model) noexcept;
    CameraParameterMask supportedCameraParameters(const FrameCamera& camera) noexcept;
    bool applyCameraParameterBlock(FrameCamera* camera,
                                   const CameraParameterBlock& parameters,
                                   const CameraParameterMask& mask,
                                   std::string* error = nullptr) noexcept;
    bool applyMetashapeFrameCalibration(FrameCamera* camera,
                                        const MetashapeFrameCalibration& calibration,
                                        std::string* error = nullptr) noexcept;
    bool metashapeFrameCalibration(const FrameCamera& camera,
                                   MetashapeFrameCalibration* calibration,
                                   std::string* error = nullptr) noexcept;
    bool cameraAtLine(const FrameCamera& camera, double linePixels, FrameCamera* effectiveCamera) noexcept;
    std::array<double, 3> worldToCameraPoint(const FrameCamera& camera, const std::array<double, 3>& world) noexcept;
    std::array<double, 3>
    worldToCameraPointAtLine(const FrameCamera& camera, const std::array<double, 3>& world, double linePixels) noexcept;
    double positiveDepth(const FrameCamera& camera, const std::array<double, 3>& world) noexcept;
    bool
    projectWorldPoint(const FrameCamera& camera, const std::array<double, 3>& world, Projection* projection) noexcept;
    bool projectWorldPointAtLine(const FrameCamera& camera,
                                 const std::array<double, 3>& world,
                                 double linePixels,
                                 Projection* projection) noexcept;
    bool linearizeCameraPoint(const FrameCamera& camera,
                              const std::array<double, 3>& cameraPoint,
                              ProjectionLinearization* linearization) noexcept;
    bool applyPoseDelta(FrameCamera* camera, const std::array<double, 6>& delta, std::string* error = nullptr) noexcept;

} // namespace plabundle
