#pragma once

#include "BundleAdjustOptions.h"
#include "BundleAdjustProblem.h"

#include <array>

namespace plabundle::internal::plamatrix_ba
{

    struct ConstraintLinearization
    {
        int residualSize = 0;
        std::array<double, 6> residual{};
        std::array<double, 6 * kBAIntrinsicParameterCount> primaryJacobian{};
        std::array<double, 6 * kBAIntrinsicParameterCount> secondaryPrimaryJacobian{};
        std::array<double, 18> pointJacobian{};
        double normalWeight = 1.0;
        double robustCost = 0.0;
    };

    bool linearizeLaserPlane(const BALaserPlaneConstraint& constraint,
                             const std::array<double, 3>& point,
                             const BAOptions& options,
                             ConstraintLinearization* output);

    bool linearizeControlPoint(const BAControlPointConstraint& constraint,
                               const std::array<double, 3>& point,
                               const BAOptions& options,
                               ConstraintLinearization* output);

    bool linearizeScaleBar(const BAScaleBarConstraint& constraint,
                           const std::array<double, 3>& point_a,
                           const std::array<double, 3>& point_b,
                           const BAOptions& options,
                           ConstraintLinearization* output);

    bool linearizePosePrior(const CameraState& camera,
                            const BACameraPosePrior& prior,
                            const BAOptions& options,
                            ConstraintLinearization* output);

    bool linearizeCameraPlane(const CameraState& camera,
                              std::size_t camera_index,
                              const BAOptions& options,
                              ConstraintLinearization* output);

    bool linearizeLaserRange(const CameraState& camera,
                             const BALaserRangeConstraint& constraint,
                             const std::array<double, 3>& point,
                             const BAOptions& options,
                             ConstraintLinearization* output);

    bool linearizeLaserPointPrior(const BALaserRangeConstraint& constraint,
                                  const std::array<double, 3>& point,
                                  ConstraintLinearization* output);

} // namespace plabundle::internal::plamatrix_ba
