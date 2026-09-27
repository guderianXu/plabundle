#include "BundleAdjustPlaMatrixProjection.h"

#include "BundleAdjustValidation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace plabundle::internal::plamatrix_ba
{
    namespace
    {

        double referencePointScale(const std::array<double, 3>& point)
        {
            const double length = std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
            const double scaled_length = length / 1000.0;
            if (!std::isfinite(scaled_length) || scaled_length == 0.0)
            {
                return 1.0;
            }
            return scaled_length / std::atan(scaled_length);
        }

        bool pixelByCameraPointJacobian(const CameraState& camera, const double camera_point[3], double jacobian[6])
        {
            if (!camera_point || !jacobian)
            {
                return false;
            }
            ProjectionLinearization linearization;
            if (!linearizeCameraPoint(
                    camera.frameCamera(), {camera_point[0], camera_point[1], camera_point[2]}, &linearization))
            {
                return false;
            }
            std::copy(linearization.cameraPointJacobian.begin(), linearization.cameraPointJacobian.end(), jacobian);
            return true;
        }

        CameraState cameraWithSharedIntrinsics(const CameraState& camera,
                                               const CameraState& reference_camera,
                                               const std::array<double, kBAIntrinsicParameterCount>& parameters,
                                               const BAIntrinsicParameterMask& active)
        {
            const auto enabled = [&](BAIntrinsicParameter parameter)
            { return active[static_cast<std::size_t>(parameter)]; };
            CameraState effective = camera;
            const auto source_intrinsics = camera.intrinsics();
            const auto reference_intrinsics = reference_camera.intrinsics();
            const double focal_x =
                enabled(BAIntrinsicParameter::FocalLength) ? parameters[0] : source_intrinsics.focalX;
            const double source_aspect =
                source_intrinsics.focalX > 1e-12 ? source_intrinsics.focalY / source_intrinsics.focalX : 1.0;
            const double aspect =
                enabled(BAIntrinsicParameter::FocalAspectRatio) ? std::exp(parameters[1]) : source_aspect;
            const double principal_x = enabled(BAIntrinsicParameter::PrincipalPointX)
                                           ? reference_intrinsics.principalX + parameters[2]
                                           : source_intrinsics.principalX;
            const double principal_y = enabled(BAIntrinsicParameter::PrincipalPointY)
                                           ? reference_intrinsics.principalY + parameters[3]
                                           : source_intrinsics.principalY;
            effective.setIntrinsics(focal_x, focal_x * aspect, principal_x, principal_y);

            auto parameter_block = effective.parameterBlock();
            CameraParameterMask parameter_mask{};
            const CameraParameterMask supported = effective.supportedParameters();
            for (std::size_t index = 4; index < parameter_block.size(); ++index)
            {
                parameter_mask[index] = active[index] && supported[index];
                if (parameter_mask[index])
                {
                    parameter_block[index] = parameters[index];
                }
            }
            (void)effective.setParameterBlock(parameter_block, parameter_mask);
            return effective;
        }

    } // namespace

    ImageRobustLossEvaluation
    evaluateImageRobustLoss(double squared_residual_norm, ImageRobustLoss loss, double scale_pixels)
    {
        if (!std::isfinite(squared_residual_norm) || squared_residual_norm < 0.0)
        {
            throw std::invalid_argument("image robust loss requires a finite non-negative squared residual norm");
        }
        if (!std::isfinite(scale_pixels) || scale_pixels <= 0.0)
        {
            throw std::invalid_argument("image robust loss scale must be finite and positive");
        }

        switch (loss)
        {
        case ImageRobustLoss::LeastSquares:
            return {0.5 * squared_residual_norm, 1.0};
        case ImageRobustLoss::Huber:
        {
            const double squared_scale = scale_pixels * scale_pixels;
            if (squared_residual_norm <= squared_scale)
            {
                return {0.5 * squared_residual_norm, 1.0};
            }
            const double residual_norm = std::sqrt(squared_residual_norm);
            return {scale_pixels * residual_norm - 0.5 * squared_scale, scale_pixels / residual_norm};
        }
        case ImageRobustLoss::Cauchy:
        {
            const double squared_scale = scale_pixels * scale_pixels;
            const double ratio = squared_residual_norm / squared_scale;
            return {0.5 * squared_scale * std::log1p(ratio), 1.0 / (1.0 + ratio)};
        }
        }
        throw std::invalid_argument("image robust loss value is invalid");
    }

    bool linearizeObservation(const CameraState& camera,
                              const std::array<double, 3>& point,
                              const BAObservation& observation,
                              ImageRobustLoss robust_loss,
                              double robust_loss_scale_pixels,
                              ObservationLinearization* linearization,
                              bool whiten_by_measurement_scale,
                              bool use_reference_point_parameterization)
    {
        if (!linearization || !observationDataIsUsable(observation))
        {
            return false;
        }
        const double world[3] = {point[0], point[1], point[2]};
        double pixel[2] = {0.0, 0.0};
        double positive_depth = 0.0;
        if (!camera.projectWorldPointWithDepthAtLine(world, observation.v, pixel, positive_depth))
        {
            return false;
        }
        double camera_point[3] = {0.0, 0.0, 0.0};
        camera.worldToCameraAtLine(world, observation.v, camera_point);
        double pixel_by_camera[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (!pixelByCameraPointJacobian(camera, camera_point, pixel_by_camera))
        {
            return false;
        }
        FrameCamera effective_frame;
        if (!cameraAtLine(camera.frameCamera(), observation.v, &effective_frame))
        {
            return false;
        }
        const CameraState effective_pose(std::move(effective_frame));

        *linearization = ObservationLinearization{};
        linearization->residual = {{pixel[0] - observation.u, pixel[1] - observation.v}};
        const auto rotation = effective_pose.cameraToWorldRotation();
        for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
        {
            for (int world_axis = 0; world_axis < 3; ++world_axis)
            {
                double derivative = 0.0;
                for (int camera_axis = 0; camera_axis < 3; ++camera_axis)
                {
                    derivative +=
                        pixel_by_camera[pixel_axis * 3 + camera_axis] * rotation[world_axis * 3 + camera_axis];
                }
                linearization->pointJacobian[pixel_axis * 3 + world_axis] = derivative;
                linearization->cameraJacobian[pixel_axis * 6 + 3 + world_axis] = -derivative;
            }

            const double* point_jacobian = linearization->pointJacobian.data() + pixel_axis * 3;
            const double dx = point[0] - effective_pose.cameraCenter()[0];
            const double dy = point[1] - effective_pose.cameraCenter()[1];
            const double dz = point[2] - effective_pose.cameraCenter()[2];
            linearization->cameraJacobian[pixel_axis * 6 + 0] = point_jacobian[1] * dz - point_jacobian[2] * dy;
            linearization->cameraJacobian[pixel_axis * 6 + 1] = -point_jacobian[0] * dz + point_jacobian[2] * dx;
            linearization->cameraJacobian[pixel_axis * 6 + 2] = point_jacobian[0] * dy - point_jacobian[1] * dx;

            if (use_reference_point_parameterization)
            {
                // 参考 type-4 姿态在转换到 CV camera-to-world 约定后等价于
                // R * D * Rx(x) * Ry(y) * Rz(z) * D，D=diag(1,-1,-1)。
                // 因此 y/z 局部轴相对普通右扰动反号。
                const double* camera_jacobian = pixel_by_camera + pixel_axis * 3;
                linearization->cameraJacobian[pixel_axis * 6 + 0] =
                    camera_jacobian[1] * camera_point[2] - camera_jacobian[2] * camera_point[1];
                linearization->cameraJacobian[pixel_axis * 6 + 1] =
                    camera_jacobian[0] * camera_point[2] - camera_jacobian[2] * camera_point[0];
                linearization->cameraJacobian[pixel_axis * 6 + 2] =
                    -camera_jacobian[0] * camera_point[1] + camera_jacobian[1] * camera_point[0];
            }
        }

        if (camera.projectionModel() == FrameProjectionModel::RollingShutter && !use_reference_point_parameterization)
        {
            const auto base_rotation = camera.cameraToWorldRotation();
            const auto effective_rotation = effective_pose.cameraToWorldRotation();
            std::array<double, 9> rolling_rotation{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int inner = 0; inner < 3; ++inner)
                    {
                        rolling_rotation[row * 3 + column] +=
                            effective_rotation[row * 3 + inner] * base_rotation[column * 3 + inner];
                    }
                }
            }
            for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
            {
                const std::array<double, 3> effective_derivative{{linearization->cameraJacobian[pixel_axis * 6],
                                                                  linearization->cameraJacobian[pixel_axis * 6 + 1],
                                                                  linearization->cameraJacobian[pixel_axis * 6 + 2]}};
                for (int base_axis = 0; base_axis < 3; ++base_axis)
                {
                    linearization->cameraJacobian[pixel_axis * 6 + base_axis] =
                        effective_derivative[0] * rolling_rotation[base_axis] +
                        effective_derivative[1] * rolling_rotation[3 + base_axis] +
                        effective_derivative[2] * rolling_rotation[6 + base_axis];
                }
            }
        }

        if (use_reference_point_parameterization)
        {
            const double point_scale = referencePointScale(point);
            for (double& value : linearization->pointJacobian)
            {
                value *= point_scale;
            }
        }

        double observation_weight = sanitizedObservationWeight(observation);
        if (whiten_by_measurement_scale)
        {
            const double measurement_scale = sanitizedMeasurementScale(observation);
            observation_weight /= measurement_scale * measurement_scale;
        }
        const double squared_norm = observation_weight * (linearization->residual[0] * linearization->residual[0] +
                                                          linearization->residual[1] * linearization->residual[1]);
        ImageRobustLossEvaluation robust;
        try
        {
            robust = evaluateImageRobustLoss(squared_norm, robust_loss, robust_loss_scale_pixels);
        }
        catch (const std::invalid_argument&)
        {
            return false;
        }
        linearization->normalWeight = observation_weight * robust.weight;
        linearization->robustCost = robust.cost;
        return std::isfinite(linearization->normalWeight) && std::isfinite(linearization->robustCost);
    }

    bool
    linearizeObservationWithSharedIntrinsics(const CameraState& camera,
                                             const CameraState& reference_camera,
                                             const std::array<double, kBAIntrinsicParameterCount>& shared_intrinsics,
                                             const BAIntrinsicParameterMask& active_parameters,
                                             const std::array<double, 3>& point,
                                             const BAObservation& observation,
                                             ImageRobustLoss robust_loss,
                                             double robust_loss_scale_pixels,
                                             ObservationLinearization* linearization,
                                             bool whiten_by_measurement_scale,
                                             bool use_reference_point_parameterization)
    {
        if (!linearization || !std::all_of(shared_intrinsics.begin(),
                                           shared_intrinsics.end(),
                                           [](double value) { return std::isfinite(value); }))
        {
            return false;
        }
        const CameraState effective =
            cameraWithSharedIntrinsics(camera, reference_camera, shared_intrinsics, active_parameters);
        if (!linearizeObservation(effective,
                                  point,
                                  observation,
                                  robust_loss,
                                  robust_loss_scale_pixels,
                                  linearization,
                                  whiten_by_measurement_scale,
                                  use_reference_point_parameterization))
        {
            return false;
        }

        const double world[3] = {point[0], point[1], point[2]};
        double camera_point[3] = {0.0, 0.0, 0.0};
        effective.worldToCameraAtLine(world, observation.v, camera_point);
        ProjectionLinearization model_linearization;
        if (!linearizeCameraPoint(
                effective.frameCamera(), {camera_point[0], camera_point[1], camera_point[2]}, &model_linearization))
        {
            return false;
        }
        auto& jacobian = linearization->intrinsicJacobian;
        const CameraParameterMask supported = effective.supportedParameters();
        for (std::size_t parameter = 0; parameter < supported.size(); ++parameter)
        {
            if (active_parameters[parameter] && supported[parameter])
            {
                jacobian[parameter] = model_linearization.parameterJacobian[parameter];
                jacobian[kBAIntrinsicParameterCount + parameter] =
                    model_linearization.parameterJacobian[kCameraParameterCount + parameter];
            }
        }
        return std::all_of(jacobian.begin(), jacobian.end(), [](double value) { return std::isfinite(value); });
    }

} // namespace plabundle::internal::plamatrix_ba
