#include "CameraState.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace plabundle::internal
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

        std::shared_ptr<const placamera::FramePinholeDefinition>
        makeDefinition(const placamera::FramePinholeNumericState& state)
        {
            return placamera::FramePinholeDefinition::create(state.definitionId(),
                                                             state.intrinsics(),
                                                             state.distortion(),
                                                             state.pixelConvention(),
                                                             state.groundFrame(),
                                                             state.depthAxisFlipped(),
                                                             state.projectionModel(),
                                                             state.sensorMount());
        }

        placamera::FramePinholeModel makeModel(const placamera::FramePinholeNumericState& state)
        {
            return placamera::FramePinholeModel::create(state.instanceId(),
                                                        state.imageId(),
                                                        makeDefinition(state),
                                                        state.imageSize(),
                                                        state.pose(),
                                                        state.captureTime(),
                                                        state.acquisition());
        }

        std::array<double, 3> toCameraPoint(const placamera::Pose& pose,
                                            const std::array<double, 3>& world) noexcept
        {
            const double dx = world[0] - pose.center[0];
            const double dy = world[1] - pose.center[1];
            const double dz = world[2] - pose.center[2];
            return {{pose.cameraToWorldRotation[0] * dx + pose.cameraToWorldRotation[3] * dy +
                         pose.cameraToWorldRotation[6] * dz,
                     pose.cameraToWorldRotation[1] * dx + pose.cameraToWorldRotation[4] * dy +
                         pose.cameraToWorldRotation[7] * dz,
                     pose.cameraToWorldRotation[2] * dx + pose.cameraToWorldRotation[5] * dy +
                         pose.cameraToWorldRotation[8] * dz}};
        }

        bool supportedModel(placamera::FrameProjectionModel model) noexcept
        {
            return model == placamera::FrameProjectionModel::Perspective ||
                   model == placamera::FrameProjectionModel::Fisheye ||
                   model == placamera::FrameProjectionModel::EquidistantFisheye ||
                   model == placamera::FrameProjectionModel::EquisolidFisheye;
        }

        void copyParameterDerivative(const placamera::BundleProjectionLinearization& source,
                                     std::size_t source_column,
                                     double* target)
        {
            const std::size_t count = source.layout.parameterCount();
            target[0] = source.parameterJacobian[source_column];
            target[kCameraParameterCount] = source.parameterJacobian[count + source_column];
        }

    } // namespace

    CameraState::CameraState(placamera::FramePinholeNumericState camera) : _camera(std::move(camera))
    {
    }

    bool CameraState::validateNumericalState(std::string* error) const noexcept
    {
        const auto fail = [error](const char* message)
        {
            if (error)
            {
                *error = message;
            }
            return false;
        };
        if (!_camera.imageSize().isValid())
        {
            return fail("camera image size must be positive");
        }
        if (!finiteArray(_camera.pose().center) || !finiteArray(_camera.pose().cameraToWorldRotation))
        {
            return fail("camera pose must be finite");
        }
        const auto intrinsics = _camera.intrinsics();
        if (!finite(intrinsics.focalX) || !finite(intrinsics.focalY) || !finite(intrinsics.principalX) ||
            !finite(intrinsics.principalY) || !finite(intrinsics.skew) || intrinsics.focalX <= 0.0 ||
            intrinsics.focalY <= 0.0)
        {
            return fail("camera intrinsics must be finite and positive");
        }
        return true;
    }

    bool CameraState::isValid() const noexcept
    {
        return validateNumericalState();
    }

    std::optional<placamera::ImageSize> CameraState::imageSize() const noexcept
    {
        return _camera.imageSize();
    }

    CameraState::Intrinsics CameraState::intrinsics() const noexcept
    {
        const auto value = _camera.intrinsics();
        return {value.focalX, value.focalY, value.principalX, value.principalY,
                value.pixelPitch, value.uAxisSign, value.vAxisSign, value.skew};
    }

    CameraState::Distortion CameraState::distortion() const noexcept
    {
        const auto value = _camera.distortion();
        return {value.radialK1, value.radialK2, value.radialK3, value.tangentialP1, value.tangentialP2,
                value.radialK4, value.tangentialP3, value.tangentialP4};
    }

    std::array<double, 9> CameraState::cameraToWorldRotation() const noexcept
    {
        return _camera.pose().cameraToWorldRotation;
    }

    std::array<double, 3> CameraState::cameraCenter() const noexcept
    {
        return _camera.pose().center;
    }

    double CameraState::focalX() const noexcept { return _camera.intrinsics().focalX; }
    double CameraState::focalY() const noexcept { return _camera.intrinsics().focalY; }
    double CameraState::principalX() const noexcept { return _camera.intrinsics().principalX; }
    double CameraState::principalY() const noexcept { return _camera.intrinsics().principalY; }
    int CameraState::uAxisSign() const noexcept { return _camera.intrinsics().uAxisSign; }
    int CameraState::vAxisSign() const noexcept { return _camera.intrinsics().vAxisSign; }
    bool CameraState::depthAxisFlipped() const noexcept { return _camera.depthAxisFlipped(); }
    placamera::FrameProjectionModel CameraState::projectionModel() const noexcept { return _camera.projectionModel(); }
    bool CameraState::rollingShutterEnabled() const noexcept
    {
        return _camera.acquisition().rollingShutterMode != placamera::RollingShutterMode::Disabled;
    }
    placamera::BrownTangentialConvention CameraState::tangentialConvention() const noexcept
    {
        return _camera.distortion().tangentialConvention;
    }
    double CameraState::skewPixels() const noexcept { return _camera.intrinsics().skew; }

    CameraParameterBlock CameraState::parameterBlock() const noexcept
    {
        const auto calibration = _camera.calibration();
        const auto value = _camera.intrinsics();
        const auto distortion = _camera.distortion();
        CameraParameterBlock parameters{};
        parameters[0] = value.focalX;
        parameters[1] = value.focalX > 0.0 && value.focalY > 0.0 ? std::log(value.focalY / value.focalX)
                                                                 : std::numeric_limits<double>::quiet_NaN();
        parameters[2] = value.principalX;
        parameters[3] = value.principalY;
        parameters[4] = distortion.radialK1;
        parameters[5] = distortion.radialK2;
        parameters[6] = distortion.radialK3;
        parameters[7] = distortion.tangentialP1;
        parameters[8] = distortion.tangentialP2;
        parameters[9] = calibration.b2;
        parameters[10] = distortion.radialK4;
        parameters[11] = distortion.tangentialP3;
        parameters[12] = distortion.tangentialP4;
        return parameters;
    }

    CameraParameterMask CameraState::supportedParameters() const noexcept
    {
        CameraParameterMask mask{};
        mask[0] = mask[1] = mask[2] = mask[3] = true;
        if (supportedModel(_camera.projectionModel()))
        {
            std::fill(mask.begin() + 4, mask.begin() + 9, true);
        }
        if (_camera.projectionModel() == placamera::FrameProjectionModel::Perspective)
        {
            mask[9] = mask[10] = true;
            if (_camera.distortion().tangentialConvention == placamera::BrownTangentialConvention::Metashape)
            {
                mask[11] = mask[12] = true;
            }
        }
        return mask;
    }

    bool CameraState::projectWorldPoint(const double world[3], double pixel[2]) const noexcept
    {
        double depth = 0.0;
        return projectWorldPointWithDepth(world, pixel, depth);
    }

    bool CameraState::projectWorldPointWithDepth(const double world[3], double pixel[2], double& depth) const noexcept
    {
        if (!world || !pixel)
        {
            return false;
        }
        const placamera::GroundCoordinate ground{_camera.groundFrame(), {world[0], world[1], world[2]}};
        const auto projected = _camera.groundToImage(ground);
        if (!projected)
        {
            return false;
        }
        pixel[0] = projected.value().image.sample;
        pixel[1] = projected.value().image.line;
        depth = projected.value().positiveDepth.value_or(0.0);
        return true;
    }

    bool CameraState::projectWorldPointWithDepthAtLine(const double world[3],
                                                       double linePixels,
                                                       double pixel[2],
                                                       double& depth) const noexcept
    {
        if (!world || !pixel)
        {
            return false;
        }
        const placamera::GroundCoordinate ground{_camera.groundFrame(), {world[0], world[1], world[2]}};
        const auto projected = _camera.groundToImageAtLine(ground, linePixels);
        if (!projected)
        {
            return false;
        }
        pixel[0] = projected.value().image.sample;
        pixel[1] = projected.value().image.line;
        depth = projected.value().positiveDepth.value_or(0.0);
        return true;
    }

    void CameraState::worldToCamera(const double world[3], double cameraPoint[3]) const noexcept
    {
        if (!world || !cameraPoint)
        {
            return;
        }
        const auto point = toCameraPoint(_camera.pose(), {world[0], world[1], world[2]});
        std::copy(point.begin(), point.end(), cameraPoint);
    }

    void CameraState::worldToCameraAtLine(const double world[3], double, double cameraPoint[3]) const noexcept
    {
        worldToCamera(world, cameraPoint);
    }

    void CameraState::setPose(const std::array<double, 9>& rotation, const std::array<double, 3>& center) noexcept
    {
        _camera.setPose(placamera::Pose::create(_camera.groundFrame(), center, rotation));
    }

    void CameraState::setCameraCenter(const std::array<double, 3>& center) noexcept
    {
        setPose(cameraToWorldRotation(), center);
    }

    void CameraState::setIntrinsics(double focalX, double focalY, double principalX, double principalY) noexcept
    {
        auto value = _camera.intrinsics();
        value.focalX = focalX;
        value.focalY = focalY;
        value.principalX = principalX;
        value.principalY = principalY;
        _camera.setIntrinsics(value);
    }

    void CameraState::setImageSize(placamera::ImageSize imageSize) noexcept
    {
        // Image size belongs to the immutable binding. Current BA input already has a valid size.
        (void)imageSize;
    }

    void CameraState::setAxisDirections(int uDirection, int vDirection) noexcept
    {
        auto value = _camera.intrinsics();
        value.uAxisSign = uDirection;
        value.vAxisSign = vDirection;
        _camera.setIntrinsics(value);
    }

    void CameraState::setDistortion(const Distortion& distortion) noexcept
    {
        auto value = _camera.distortion();
        value.radialK1 = distortion.radialK1;
        value.radialK2 = distortion.radialK2;
        value.radialK3 = distortion.radialK3;
        value.tangentialP1 = distortion.tangentialP1;
        value.tangentialP2 = distortion.tangentialP2;
        value.radialK4 = distortion.radialK4;
        value.tangentialP3 = distortion.tangentialP3;
        value.tangentialP4 = distortion.tangentialP4;
        _camera.setDistortion(value);
    }

    void CameraState::setDistortion(double k1, double k2, double k3, double p1, double p2) noexcept
    {
        auto value = distortion();
        value.radialK1 = k1;
        value.radialK2 = k2;
        value.radialK3 = k3;
        value.tangentialP1 = p1;
        value.tangentialP2 = p2;
        setDistortion(value);
    }

    bool CameraState::setParameterBlock(const CameraParameterBlock& parameters,
                                        const CameraParameterMask& mask,
                                        std::string* error) noexcept
    {
        if (!std::all_of(parameters.begin(), parameters.end(), finite))
        {
            if (error)
            {
                *error = "camera parameter block must contain finite values";
            }
            return false;
        }
        const auto supported = supportedParameters();
        for (std::size_t index = 0; index < mask.size(); ++index)
        {
            if (mask[index] && !supported[index])
            {
                if (error)
                {
                    *error = "camera parameter mask enables an unsupported parameter";
                }
                return false;
            }
        }
        CameraParameterBlock effective = parameterBlock();
        for (std::size_t index = 0; index < mask.size(); ++index)
        {
            if (mask[index])
            {
                effective[index] = parameters[index];
            }
        }
        setIntrinsics(effective[0], effective[0] * std::exp(effective[1]), effective[2], effective[3]);
        auto intrinsics = _camera.intrinsics();
        intrinsics.skew = effective[9];
        _camera.setIntrinsics(intrinsics);
        auto distortion = _camera.distortion();
        distortion.radialK1 = effective[4];
        distortion.radialK2 = effective[5];
        distortion.radialK3 = effective[6];
        distortion.tangentialP1 = effective[7];
        distortion.tangentialP2 = effective[8];
        distortion.radialK4 = effective[10];
        distortion.tangentialP3 = effective[11];
        distortion.tangentialP4 = effective[12];
        _camera.setDistortion(distortion);
        return validateNumericalState(error);
    }

    void CameraState::applyDeltaPose(const double delta[6]) noexcept
    {
        if (delta)
        {
            _camera.applyPoseDelta({delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]});
        }
    }

    CameraState CameraState::normalizedForPositiveDepth() const noexcept
    {
        return CameraState(_camera.normalizedForPositiveDepth());
    }

    bool CameraState::linearize(const std::array<double, 3>& point,
                                double,
                                bool includeParameters,
                                ProjectionLinearization* output) const noexcept
    {
        if (!output)
        {
            return false;
        }
        try
        {
            const auto model = makeModel(_camera);
            const placamera::GroundCoordinate ground{_camera.groundFrame(), point};
            const auto linearization = placamera::linearizeForBundle(model, ground);
            if (!linearization)
            {
                return false;
            }
            const auto& source = linearization.value();
            output->projection.pixel = {{source.projection.image.sample, source.projection.image.line}};
            output->projection.positiveDepth = source.projection.positiveDepth.value_or(0.0);
            const auto rotation = _camera.pose().cameraToWorldRotation;
            for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
            {
                for (int camera_axis = 0; camera_axis < 3; ++camera_axis)
                {
                    double derivative = 0.0;
                    for (int world_axis = 0; world_axis < 3; ++world_axis)
                    {
                        derivative += source.pointJacobian[pixel_axis * 3 + world_axis] *
                                      rotation[world_axis * 3 + camera_axis];
                    }
                    output->cameraPointJacobian[pixel_axis * 3 + camera_axis] = derivative;
                }
            }
            output->parameterJacobian.fill(0.0);
            if (!includeParameters)
            {
                return true;
            }
            const auto calibration = _camera.calibration();
            const double aspect = _camera.intrinsics().focalX > 0.0
                                      ? _camera.intrinsics().focalY / _camera.intrinsics().focalX
                                      : 1.0;
            double derivativeF[2] = {0.0, 0.0};
            double derivativeB1[2] = {0.0, 0.0};
            for (const auto& block : source.layout.blocks)
            {
                for (std::size_t component = 0; component < block.size; ++component)
                {
                    const std::size_t column = block.offset + component;
                    double sample = source.parameterJacobian[column];
                    double line = source.parameterJacobian[source.layout.parameterCount() + column];
                    auto assign = [&](std::size_t target)
                    {
                        output->parameterJacobian[target] = sample;
                        output->parameterJacobian[kCameraParameterCount + target] = line;
                    };
                    if (block.name == "calibration.f")
                    {
                        derivativeF[0] += sample;
                        derivativeF[1] += line;
                    }
                    else if (block.name == "calibration.b1")
                    {
                        derivativeB1[0] += sample;
                        derivativeB1[1] += line;
                    }
                    else if (block.name == "calibration.cx") assign(2);
                    else if (block.name == "calibration.cy") assign(3);
                    else if (block.name == "calibration.b2") assign(9);
                    else if (block.name == "distortion.k1") assign(4);
                    else if (block.name == "distortion.k2") assign(5);
                    else if (block.name == "distortion.k3") assign(6);
                    else if (block.name == "distortion.k4") assign(10);
                    else if (block.name == "distortion.p1") assign(7);
                    else if (block.name == "distortion.p2") assign(8);
                    else if (block.name == "distortion.p3") assign(11);
                    else if (block.name == "distortion.p4") assign(12);
                }
            }
            output->parameterJacobian[0] = aspect * derivativeF[0] + (1.0 - aspect) * derivativeB1[0];
            output->parameterJacobian[kCameraParameterCount] =
                aspect * derivativeF[1] + (1.0 - aspect) * derivativeB1[1];
            const double focalY = _camera.intrinsics().focalY;
            output->parameterJacobian[1] = focalY * (derivativeF[0] - derivativeB1[0]);
            output->parameterJacobian[kCameraParameterCount + 1] = focalY * (derivativeF[1] - derivativeB1[1]);
            (void)calibration;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    const placamera::FramePinholeNumericState& CameraState::nativeState() const noexcept { return _camera; }
    placamera::FramePinholeNumericState& CameraState::nativeState() noexcept { return _camera; }

    std::vector<CameraState> makeCameraStates(const std::vector<placamera::FramePinholeNumericState>& cameras)
    {
        std::vector<CameraState> states;
        states.reserve(cameras.size());
        for (const auto& camera : cameras)
        {
            states.emplace_back(camera);
        }
        return states;
    }

    std::vector<placamera::FramePinholeNumericState>
    makeNumericStates(const std::vector<CameraState>& cameras,
                      const std::vector<placamera::FramePinholeNumericState>& templates)
    {
        if (cameras.size() != templates.size())
        {
            return {};
        }
        std::vector<placamera::FramePinholeNumericState> states;
        states.reserve(cameras.size());
        for (const auto& camera : cameras)
        {
            states.push_back(camera.nativeState());
        }
        return states;
    }

} // namespace plabundle::internal
