#include "internal/BundleAdjustPlaMatrixConstraints.h"
#include "internal/CameraState.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace
{
    plabundle::internal::CameraState makeCamera(double centerX, double centerY)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter = {centerX, centerY, 0.0};
        camera.focalXPixels = 960.0;
        camera.focalYPixels = 950.0;
        camera.principalXPixel = 512.0;
        camera.principalYPixel = 384.0;
        camera.distortion = {-0.012, 0.0008, -0.00005, 0.0002, -0.00015};
        return plabundle::internal::CameraState(camera);
    }

} // namespace

TEST(PlaBundleConstraintJacobianTest, PosePriorAndLaserRangeMatchCentralDifferences)
{
    namespace solver = plabundle::internal::plamatrix_ba;
    using plabundle::internal::BACameraPosePrior;
    using plabundle::internal::BALaserRangeConstraint;
    using plabundle::internal::BAOptions;

    auto camera = makeCamera(0.4, -0.7);
    const double initial_delta[6] = {0.03, -0.02, 0.01, 0.0, 0.0, 0.0};
    camera.applyDeltaPose(initial_delta);

    BAOptions options;
    options.cameraPosePriorWeight = 3.5;
    options.cameraPosePriorHuberDelta = 0.0;
    options.laserRangeWeight = 2.5;
    options.laserRangeHuberDelta = 0.0;
    BACameraPosePrior prior;
    prior.enabled = true;
    prior.cameraCenter = {0.3, -0.5, 0.2};
    prior.cameraToWorldRotation = makeCamera(0.0, 0.0).cameraToWorldRotation();
    prior.positionSigmaMeters = 0.7;
    prior.rotationSigmaDegrees = 2.0;

    solver::ConstraintLinearization pose;
    ASSERT_TRUE(solver::linearizePosePrior(camera, prior, options, &pose));
    constexpr double epsilon = 1.0e-7;
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        double delta[6]{};
        delta[parameter] = epsilon;
        auto plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        auto minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        solver::ConstraintLinearization plus;
        solver::ConstraintLinearization minus;
        ASSERT_TRUE(solver::linearizePosePrior(plus_camera, prior, options, &plus));
        ASSERT_TRUE(solver::linearizePosePrior(minus_camera, prior, options, &minus));
        for (int row = 0; row < pose.residualSize; ++row)
        {
            const double numerical =
                (plus.residual[static_cast<std::size_t>(row)] - minus.residual[static_cast<std::size_t>(row)]) /
                (2.0 * epsilon);
            EXPECT_NEAR(pose.primaryJacobian[static_cast<std::size_t>(row * 9 + parameter)], numerical, 2.0e-5);
        }
    }

    BALaserRangeConstraint shot;
    shot.cameraIndex = 0;
    shot.observedRangeMeters = 9.2;
    shot.sigmaRangeMeters = 0.4;
    shot.weight = 1.3;
    shot.leverArmCameraMeters = {0.3, -0.15, 0.08};
    const std::array<double, 3> point{{1.1, -0.2, 9.8}};
    solver::ConstraintLinearization range;
    ASSERT_TRUE(solver::linearizeLaserRange(camera, shot, point, options, &range));
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        double delta[6]{};
        delta[parameter] = epsilon;
        auto plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        auto minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        solver::ConstraintLinearization plus;
        solver::ConstraintLinearization minus;
        ASSERT_TRUE(solver::linearizeLaserRange(plus_camera, shot, point, options, &plus));
        ASSERT_TRUE(solver::linearizeLaserRange(minus_camera, shot, point, options, &minus));
        const double numerical = (plus.residual[0] - minus.residual[0]) / (2.0 * epsilon);
        EXPECT_NEAR(range.primaryJacobian[static_cast<std::size_t>(parameter)], numerical, 2.0e-6);
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        auto plus_point = point;
        auto minus_point = point;
        plus_point[static_cast<std::size_t>(axis)] += epsilon;
        minus_point[static_cast<std::size_t>(axis)] -= epsilon;
        solver::ConstraintLinearization plus;
        solver::ConstraintLinearization minus;
        ASSERT_TRUE(solver::linearizeLaserRange(camera, shot, plus_point, options, &plus));
        ASSERT_TRUE(solver::linearizeLaserRange(camera, shot, minus_point, options, &minus));
        const double numerical = (plus.residual[0] - minus.residual[0]) / (2.0 * epsilon);
        EXPECT_NEAR(range.pointJacobian[static_cast<std::size_t>(axis)], numerical, 2.0e-6);
    }
}
