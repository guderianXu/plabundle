#pragma once

/**
 * @file BundleAdjustPlaMatrixProjection.h
 * @brief PlaMatrix BA 使用的固定/共享 Brown 内参重投影解析线性化。
 */

#include "BundleAdjustProblem.h"
#include "BundleAdjustTypes.h"
#include "CameraState.h"

#include <array>

namespace plabundle::internal::plamatrix_ba
{

    /// 一条二维重投影观测的原始残差、解析雅可比和鲁棒法方程权重。
    struct ObservationLinearization
    {
        std::array<double, 2> residual{{0.0, 0.0}};
        std::array<double, 6> pointJacobian{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        std::array<double, 12> cameraJacobian{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        std::array<double, 2 * kBAIntrinsicParameterCount> intrinsicJacobian{};
        double normalWeight = 1.0;
        double robustCost = 0.0;
    };

    struct ImageRobustLossEvaluation
    {
        double cost = 0.0;
        double weight = 1.0;
    };

    ImageRobustLossEvaluation
    evaluateImageRobustLoss(double squared_residual_norm, ImageRobustLoss loss, double scale_pixels);

    /// Evaluate only the robust reprojection cost, without forming Jacobians.
    bool evaluateObservationCost(const CameraState& camera,
                                 const std::array<double, 3>& point,
                                 const BAObservation& observation,
                                 ImageRobustLoss robust_loss,
                                 double robust_loss_scale_pixels,
                                 double* cost,
                                 bool whiten_by_measurement_scale = false);

    /// Cost-only counterpart of linearizeObservationWithSharedIntrinsics.
    bool
    evaluateObservationCostWithSharedIntrinsics(const CameraState& camera,
                                                const CameraState& reference_camera,
                                                const std::array<double, kBAIntrinsicParameterCount>& shared_intrinsics,
                                                const BAIntrinsicParameterMask& active_parameters,
                                                const std::array<double, 3>& point,
                                                const BAObservation& observation,
                                                ImageRobustLoss robust_loss,
                                                double robust_loss_scale_pixels,
                                                double* cost,
                                                bool whiten_by_measurement_scale = false);

    /// Build the per-camera numerical state for one shared-intrinsic stage.
    CameraState cameraWithSharedIntrinsics(const CameraState& camera,
                                           const CameraState& reference_camera,
                                           const std::array<double, kBAIntrinsicParameterCount>& shared_intrinsics,
                                           const BAIntrinsicParameterMask& active_parameters);

    /**
     * @brief 按 FramePinholeNumericState 投影语义线性化单条观测。
     *
     * cameraJacobian 对应局部参数 `[wx, wy, wz, dCx, dCy, dCz]`，旋转增量左乘
     * camera-to-world 旋转，中心增量位于世界坐标系。返回 false 表示点不在物理前方、
     * 输入非法或投影/雅可比非有限。
     * model_linearization 可选地接收同一次相机点投影导数，供共享内参雅可比复用。
     */
    bool linearizeObservation(const CameraState& camera,
                              const std::array<double, 3>& point,
                              const BAObservation& observation,
                              ImageRobustLoss robust_loss,
                              double robust_loss_scale_pixels,
                              ObservationLinearization* linearization,
                              bool whiten_by_measurement_scale = false,
                              bool use_reference_point_parameterization = false,
                              ProjectionLinearization* model_linearization = nullptr);

    /// Linearize reprojection with the shared complete frame-calibration model.
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
                                             bool whiten_by_measurement_scale = false,
                                             bool use_reference_point_parameterization = false);

    /// Shared-intrinsic linearization when the effective camera is already cached.
    bool linearizeObservationWithActiveIntrinsics(const CameraState& effective_camera,
                                                  const BAIntrinsicParameterMask& active_parameters,
                                                  const std::array<double, 3>& point,
                                                  const BAObservation& observation,
                                                  ImageRobustLoss robust_loss,
                                                  double robust_loss_scale_pixels,
                                                  ObservationLinearization* linearization,
                                                  bool whiten_by_measurement_scale = false,
                                                  bool use_reference_point_parameterization = false);

} // namespace plabundle::internal::plamatrix_ba
