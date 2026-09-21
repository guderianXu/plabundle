#include "CameraState.h"

#include <cmath>
#include <cstddef>
#include <utility>

namespace plabundle::internal
{

    CameraState::CameraState(FrameCamera camera) : _camera(std::move(camera))
    {
    }

    bool CameraState::validateNumericalState(std::string* error) const noexcept
    {
        return validateFrameCamera(_camera, error);
    }

    bool CameraState::isValid() const noexcept
    {
        return validateFrameCamera(_camera);
    }

    std::optional<ImageSize> CameraState::imageSize() const noexcept
    {
        return _camera.imageSize;
    }

    CameraState::Intrinsics CameraState::intrinsics() const noexcept
    {
        return {_camera.focalXPixels,
                _camera.focalYPixels,
                _camera.principalXPixel,
                _camera.principalYPixel,
                _camera.pixelPitchMillimeters,
                _camera.uAxisSign,
                _camera.vAxisSign};
    }

    CameraState::Distortion CameraState::distortion() const noexcept
    {
        return {_camera.distortion.k1,
                _camera.distortion.k2,
                _camera.distortion.k3,
                _camera.distortion.p1,
                _camera.distortion.p2};
    }

    std::array<double, 9> CameraState::cameraToWorldRotation() const noexcept
    {
        return _camera.cameraToWorldRotation;
    }

    std::array<double, 3> CameraState::cameraCenter() const noexcept
    {
        return _camera.cameraCenter;
    }

    double CameraState::focalX() const noexcept
    {
        return _camera.focalXPixels;
    }

    double CameraState::focalY() const noexcept
    {
        return _camera.focalYPixels;
    }

    double CameraState::principalX() const noexcept
    {
        return _camera.principalXPixel;
    }

    double CameraState::principalY() const noexcept
    {
        return _camera.principalYPixel;
    }

    int CameraState::uAxisSign() const noexcept
    {
        return _camera.uAxisSign;
    }

    int CameraState::vAxisSign() const noexcept
    {
        return _camera.vAxisSign;
    }

    bool CameraState::depthAxisFlipped() const noexcept
    {
        return _camera.depthAxisFlipped;
    }

    bool CameraState::projectWorldPoint(const double world[3], double pixel[2]) const noexcept
    {
        if (!world || !pixel)
        {
            return false;
        }
        Projection projection;
        if (!plabundle::projectWorldPoint(_camera, {world[0], world[1], world[2]}, &projection))
        {
            return false;
        }
        pixel[0] = projection.pixel[0];
        pixel[1] = projection.pixel[1];
        return true;
    }

    bool CameraState::projectWorldPointWithDepth(const double world[3], double pixel[2], double& depth) const noexcept
    {
        if (!world || !pixel)
        {
            return false;
        }
        Projection projection;
        if (!plabundle::projectWorldPoint(_camera, {world[0], world[1], world[2]}, &projection))
        {
            return false;
        }
        pixel[0] = projection.pixel[0];
        pixel[1] = projection.pixel[1];
        depth = projection.positiveDepth;
        return true;
    }

    void CameraState::worldToCamera(const double world[3], double cameraPoint[3]) const noexcept
    {
        if (!world || !cameraPoint)
        {
            return;
        }
        const std::array<double, 3> transformed = worldToCameraPoint(_camera, {world[0], world[1], world[2]});
        cameraPoint[0] = transformed[0];
        cameraPoint[1] = transformed[1];
        cameraPoint[2] = transformed[2];
    }

    void CameraState::setPose(const std::array<double, 9>& rotation, const std::array<double, 3>& center) noexcept
    {
        _camera.cameraToWorldRotation = rotation;
        _camera.cameraCenter = center;
    }

    void CameraState::setCameraCenter(const std::array<double, 3>& center) noexcept
    {
        _camera.cameraCenter = center;
    }

    void CameraState::setIntrinsics(double focalX, double focalY, double principalX, double principalY) noexcept
    {
        _camera.focalXPixels = focalX;
        _camera.focalYPixels = focalY;
        _camera.principalXPixel = principalX;
        _camera.principalYPixel = principalY;
    }

    void CameraState::setImageSize(ImageSize imageSize) noexcept
    {
        _camera.imageSize = imageSize;
    }

    void CameraState::setAxisDirections(int uDirection, int vDirection) noexcept
    {
        _camera.uAxisSign = uDirection;
        _camera.vAxisSign = vDirection;
    }

    void CameraState::setDistortion(const Distortion& distortion) noexcept
    {
        setDistortion(distortion.radialK1,
                      distortion.radialK2,
                      distortion.radialK3,
                      distortion.tangentialP1,
                      distortion.tangentialP2);
    }

    void CameraState::setDistortion(double k1, double k2, double k3, double p1, double p2) noexcept
    {
        _camera.distortion = {k1, k2, k3, p1, p2};
    }

    void CameraState::applyDeltaPose(const double delta[6]) noexcept
    {
        if (!delta)
        {
            return;
        }
        const std::array<double, 6> values{{delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]}};
        (void)applyPoseDelta(&_camera, values);
    }

    CameraState CameraState::normalizedForPositiveDepth() const noexcept
    {
        CameraState result = *this;
        const double depth_sign = _camera.depthAxisFlipped ? -1.0 : 1.0;
        const double u_sign = _camera.uAxisSign < 0 ? -1.0 : 1.0;
        const double v_sign = _camera.vAxisSign < 0 ? -1.0 : 1.0;
        std::array<double, 3> signs{depth_sign * u_sign, depth_sign * v_sign, depth_sign};
        if (signs[0] * signs[1] * signs[2] < 0.0)
        {
            signs[0] = -signs[0];
        }
        const double x_ratio_sign = signs[0] / signs[2];
        const double y_ratio_sign = signs[1] / signs[2];
        std::array<double, 9> normalized_rotation{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                normalized_rotation[static_cast<std::size_t>(row * 3 + column)] =
                    _camera.cameraToWorldRotation[static_cast<std::size_t>(row * 3 + column)] *
                    signs[static_cast<std::size_t>(column)];
            }
        }
        result._camera.cameraToWorldRotation = normalized_rotation;
        result._camera.depthAxisFlipped = false;
        result._camera.focalXPixels = std::abs(result._camera.focalXPixels);
        result._camera.focalYPixels = std::abs(result._camera.focalYPixels);
        result._camera.uAxisSign = static_cast<int>(u_sign * x_ratio_sign);
        result._camera.vAxisSign = static_cast<int>(v_sign * y_ratio_sign);
        result._camera.distortion.p1 *= y_ratio_sign;
        result._camera.distortion.p2 *= x_ratio_sign;
        return result;
    }

    const FrameCamera& CameraState::frameCamera() const noexcept
    {
        return _camera;
    }

    std::vector<CameraState> makeCameraStates(const std::vector<FrameCamera>& cameras)
    {
        std::vector<CameraState> states;
        states.reserve(cameras.size());
        for (const FrameCamera& camera : cameras)
        {
            states.emplace_back(camera);
        }
        return states;
    }

    std::vector<FrameCamera> makeFrameCameras(const std::vector<CameraState>& cameras)
    {
        std::vector<FrameCamera> states;
        states.reserve(cameras.size());
        for (const CameraState& camera : cameras)
        {
            states.push_back(camera.frameCamera());
        }
        return states;
    }

} // namespace plabundle::internal
