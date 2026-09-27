#include "internal/BundleAdjustPlaMatrixConstraints.h"
#include "internal/CameraState.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
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

    void expectPosePriorJacobian(const plabundle::internal::CameraState& camera,
                                 const plabundle::internal::BACameraPosePrior& prior,
                                 const plabundle::internal::BAOptions& options,
                                 double tolerance = 3.0e-5)
    {
        namespace solver = plabundle::internal::plamatrix_ba;
        solver::ConstraintLinearization analytic;
        ASSERT_TRUE(solver::linearizePosePrior(camera, prior, options, &analytic));
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
            ASSERT_EQ(plus.residualSize, analytic.residualSize);
            for (int row = 0; row < analytic.residualSize; ++row)
            {
                const double numerical =
                    (plus.residual[static_cast<std::size_t>(row)] - minus.residual[static_cast<std::size_t>(row)]) /
                    (2.0 * epsilon);
                EXPECT_NEAR(analytic.primaryJacobian[static_cast<std::size_t>(
                                row * plabundle::kIntrinsicParameterCount + parameter)],
                            numerical,
                            tolerance);
            }
        }
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
            EXPECT_NEAR(
                pose.primaryJacobian[static_cast<std::size_t>(row * plabundle::kIntrinsicParameterCount + parameter)],
                numerical,
                2.0e-5);
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

TEST(PlaBundleConstraintJacobianTest, PosePriorCovarianceAndSqrtInformationWhiteningMatchCentralDifferences)
{
    using plabundle::PosePriorComponents;
    using plabundle::PosePriorTangentFrame;
    using plabundle::PosePriorUncertainty;
    using plabundle::internal::BACameraPosePrior;
    using plabundle::internal::BAOptions;

    auto prior_pose = makeCamera(0.1, -0.2);
    const double prior_rotation[6] = {0.2, -0.1, 0.15, 0.0, 0.0, 0.0};
    prior_pose.applyDeltaPose(prior_rotation);
    auto camera = prior_pose;
    const double camera_delta[6] = {0.03, -0.02, 0.01, 0.4, -0.3, 0.2};
    camera.applyDeltaPose(camera_delta);

    BAOptions options;
    options.cameraPosePriorWeight = 2.3;
    options.cameraPosePriorHuberDelta = 0.0;
    BACameraPosePrior covariance;
    covariance.enabled = true;
    covariance.cameraToWorldRotation = prior_pose.cameraToWorldRotation();
    covariance.cameraCenter = prior_pose.cameraCenter();
    covariance.components = PosePriorComponents::RotationAndPosition;
    covariance.uncertainty = PosePriorUncertainty::Covariance;
    covariance.tangentFrame = PosePriorTangentFrame::PriorCamera;
    for (int index = 0; index < 6; ++index)
    {
        covariance.uncertaintyMatrix[static_cast<std::size_t>(index * 6 + index)] = 0.2 + 0.1 * index;
    }
    covariance.uncertaintyMatrix[3] = 0.04;
    covariance.uncertaintyMatrix[18] = 0.04;
    covariance.uncertaintyMatrix[10] = -0.03;
    covariance.uncertaintyMatrix[25] = -0.03;
    ASSERT_TRUE(plabundle::validateCameraPosePrior(covariance));
    expectPosePriorJacobian(camera, covariance, options);

    BACameraPosePrior sqrt_information = covariance;
    sqrt_information.uncertainty = PosePriorUncertainty::SqrtInformation;
    sqrt_information.tangentFrame = PosePriorTangentFrame::World;
    sqrt_information.uncertaintyMatrix.fill(0.0);
    for (int index = 0; index < 6; ++index)
    {
        sqrt_information.uncertaintyMatrix[static_cast<std::size_t>(index * 6 + index)] = 1.0 + 0.2 * index;
    }
    sqrt_information.uncertaintyMatrix[4] = 0.15;
    sqrt_information.uncertaintyMatrix[23] = -0.2;
    ASSERT_TRUE(plabundle::validateCameraPosePrior(sqrt_information));
    expectPosePriorJacobian(camera, sqrt_information, options);
}

TEST(PlaBundleConstraintJacobianTest, PositionCovarianceUsesMahalanobisNormAndRotationCanBeIndependent)
{
    namespace solver = plabundle::internal::plamatrix_ba;
    using plabundle::PosePriorComponents;
    using plabundle::PosePriorUncertainty;
    using plabundle::internal::BACameraPosePrior;
    using plabundle::internal::BAOptions;

    const auto camera = makeCamera(2.0, -1.0);
    BAOptions options;
    options.cameraPosePriorWeight = 1.0;
    options.cameraPosePriorHuberDelta = 0.0;
    BACameraPosePrior prior;
    prior.enabled = true;
    prior.cameraCenter = {0.0, 0.0, 0.0};
    prior.components = PosePriorComponents::Position;
    prior.uncertainty = PosePriorUncertainty::Covariance;
    prior.uncertaintyMatrix[21] = 4.0;
    prior.uncertaintyMatrix[22] = 1.0;
    prior.uncertaintyMatrix[27] = 1.0;
    prior.uncertaintyMatrix[28] = 2.0;
    prior.uncertaintyMatrix[35] = 1.0;
    solver::ConstraintLinearization position;
    ASSERT_TRUE(solver::linearizePosePrior(camera, prior, options, &position));
    ASSERT_EQ(position.residualSize, 3);
    const double expected_squared_norm = (2.0 * 4.0 - 2.0 * 2.0 * -1.0 + 4.0 * 1.0) / 7.0;
    double actual_squared_norm = 0.0;
    for (int row = 0; row < position.residualSize; ++row)
    {
        actual_squared_norm +=
            position.residual[static_cast<std::size_t>(row)] * position.residual[static_cast<std::size_t>(row)];
    }
    EXPECT_NEAR(actual_squared_norm, expected_squared_norm, 1.0e-12);
    expectPosePriorJacobian(camera, prior, options);

    prior.components = PosePriorComponents::Rotation;
    prior.uncertainty = PosePriorUncertainty::SqrtInformation;
    prior.uncertaintyMatrix.fill(0.0);
    prior.uncertaintyMatrix[0] = 3.0;
    prior.uncertaintyMatrix[7] = 2.0;
    prior.uncertaintyMatrix[14] = 1.0;
    ASSERT_TRUE(plabundle::validateCameraPosePrior(prior));
    expectPosePriorJacobian(camera, prior, options);
}

TEST(PlaBundleConstraintJacobianTest, ControlPointCovarianceWhitensResidualAndJacobian)
{
    namespace solver = plabundle::internal::plamatrix_ba;
    using plabundle::ControlPointUncertainty;
    using plabundle::internal::BAControlPointConstraint;
    using plabundle::internal::BAOptions;

    BAControlPointConstraint constraint;
    constraint.point = {1.0, -2.0, 0.5};
    constraint.weight = 1.6;
    constraint.uncertainty = ControlPointUncertainty::Covariance;
    constraint.uncertaintyMatrix = {4.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.25};
    ASSERT_TRUE(plabundle::validateControlPointConstraint(constraint));

    BAOptions options;
    options.controlPointWeight = 2.5;
    options.controlPointHuberDeltaMeters = 0.0;
    const std::array<double, 3> point{{1.3, -1.7, 0.9}};
    solver::ConstraintLinearization analytic;
    ASSERT_TRUE(solver::linearizeControlPoint(constraint, point, options, &analytic));
    ASSERT_EQ(analytic.residualSize, 3);

    constexpr double epsilon = 1.0e-7;
    for (int parameter = 0; parameter < 3; ++parameter)
    {
        auto plus_point = point;
        auto minus_point = point;
        plus_point[static_cast<std::size_t>(parameter)] += epsilon;
        minus_point[static_cast<std::size_t>(parameter)] -= epsilon;
        solver::ConstraintLinearization plus;
        solver::ConstraintLinearization minus;
        ASSERT_TRUE(solver::linearizeControlPoint(constraint, plus_point, options, &plus));
        ASSERT_TRUE(solver::linearizeControlPoint(constraint, minus_point, options, &minus));
        for (int row = 0; row < analytic.residualSize; ++row)
        {
            const double numerical =
                (plus.residual[static_cast<std::size_t>(row)] - minus.residual[static_cast<std::size_t>(row)]) /
                (2.0 * epsilon);
            EXPECT_NEAR(analytic.pointJacobian[static_cast<std::size_t>(row * 3 + parameter)], numerical, 5.0e-9);
        }
    }

    const double total_weight = options.controlPointWeight * constraint.weight;
    const std::array<double, 9> expected_inverse{
        {2.0 / 7.0, -1.0 / 7.0, 0.0, -1.0 / 7.0, 4.0 / 7.0, 0.0, 0.0, 0.0, 4.0}};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            double information = 0.0;
            for (int inner = 0; inner < 3; ++inner)
            {
                information += analytic.pointJacobian[static_cast<std::size_t>(inner * 3 + row)] *
                               analytic.pointJacobian[static_cast<std::size_t>(inner * 3 + column)];
            }
            EXPECT_NEAR(
                information / total_weight, expected_inverse[static_cast<std::size_t>(row * 3 + column)], 1.0e-12);
        }
    }
}
