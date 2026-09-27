#include "internal/BundleAdjustPlaMatrixAssembly.h"
#include "internal/BundleAdjustPlaMatrixAssemblyInternal.h"
#include "internal/BundleAdjustPlaMatrixConstraints.h"
#include "internal/BundleAdjustPlaMatrixProblem.h"
#include "internal/BundleAdjustPlaMatrixProjection.h"
#include "internal/CameraState.h"
#include "internal/ParallelFor.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace
{
    using plabundle::internal::BAIntrinsicParameterMask;
    using plabundle::internal::BAObservation;
    using plabundle::internal::BAOptions;
    using plabundle::internal::BATrack;
    using plabundle::internal::CameraState;
    namespace solver = plabundle::internal::plamatrix_ba;

    CameraState makeCamera(double centerX, double centerY)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter = {centerX, centerY, 0.0};
        camera.focalXPixels = 920.0;
        camera.focalYPixels = 910.0;
        camera.principalXPixel = 512.0;
        camera.principalYPixel = 384.0;
        camera.distortion = {-0.015, 0.001, -0.0001, 0.0004, -0.0003};
        return CameraState(camera);
    }

    std::array<double, 2> project(const CameraState& camera, const std::array<double, 3>& point)
    {
        const double world[3] = {point[0], point[1], point[2]};
        double pixel[2] = {0.0, 0.0};
        EXPECT_TRUE(camera.projectWorldPoint(world, pixel));
        return {pixel[0], pixel[1]};
    }

    double numericPointDerivative(const CameraState& camera, std::array<double, 3> point, int parameter, int pixelAxis)
    {
        constexpr double epsilon = 1.0e-6;
        point[static_cast<std::size_t>(parameter)] += epsilon;
        const auto plus = project(camera, point);
        point[static_cast<std::size_t>(parameter)] -= 2.0 * epsilon;
        const auto minus = project(camera, point);
        return (plus[static_cast<std::size_t>(pixelAxis)] - minus[static_cast<std::size_t>(pixelAxis)]) /
               (2.0 * epsilon);
    }

    double
    numericCameraDerivative(const CameraState& camera, const std::array<double, 3>& point, int parameter, int pixelAxis)
    {
        constexpr double epsilon = 1.0e-7;
        double delta[6]{};
        delta[parameter] = epsilon;
        CameraState plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        CameraState minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        const auto plus = project(plus_camera, point);
        const auto minus = project(minus_camera, point);
        return (plus[static_cast<std::size_t>(pixelAxis)] - minus[static_cast<std::size_t>(pixelAxis)]) /
               (2.0 * epsilon);
    }

    CameraState cameraWithSharedIntrinsics(const CameraState& poseCamera,
                                           const CameraState& referenceCamera,
                                           const plabundle::CameraParameterBlock& parameters)
    {
        CameraState camera = poseCamera;
        auto effective = parameters;
        effective[2] = referenceCamera.principalX() + parameters[2];
        effective[3] = referenceCamera.principalY() + parameters[3];
        EXPECT_TRUE(camera.setParameterBlock(effective, camera.supportedParameters()));
        return camera;
    }

    CameraState makeModelCamera(plabundle::FrameProjectionModel model)
    {
        plabundle::FrameCamera camera = makeCamera(-1.2, 0.7).frameCamera();
        camera.projectionModel = model;
        camera.modelCoefficients = {{-0.015, 0.001, -0.0001, 0.00004, -0.000003}};
        if (model == plabundle::FrameProjectionModel::RollingShutter)
        {
            camera.rollingShutter.secondsPerLine = 1.0e-4;
        }
        return CameraState(camera);
    }

    BATrack makeTrack(const std::vector<CameraState>& cameras, const std::array<double, 3>& point)
    {
        BATrack track;
        track.initialPoint = point;
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
        {
            const auto pixel = project(cameras[camera_index], point);
            track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
        }
        return track;
    }

    void applyRigDelta(plabundle::RigTopology* rig, bool capture, std::size_t index, int parameter, double value)
    {
        plabundle::FrameCamera pose;
        if (capture)
        {
            pose.cameraToWorldRotation = rig->captures[index].rigToWorldRotation;
            pose.cameraCenter = rig->captures[index].rigCenterInWorld;
        }
        else
        {
            pose.cameraToWorldRotation = rig->sensors[index].cameraToRigRotation;
            pose.cameraCenter = rig->sensors[index].cameraCenterInRig;
        }
        pose.focalXPixels = 1.0;
        pose.focalYPixels = 1.0;
        std::array<double, 6> delta{};
        delta[static_cast<std::size_t>(parameter)] = value;
        ASSERT_TRUE(plabundle::applyPoseDelta(&pose, delta));
        if (capture)
        {
            rig->captures[index].rigToWorldRotation = pose.cameraToWorldRotation;
            rig->captures[index].rigCenterInWorld = pose.cameraCenter;
        }
        else
        {
            rig->sensors[index].cameraToRigRotation = pose.cameraToWorldRotation;
            rig->sensors[index].cameraCenterInRig = pose.cameraCenter;
        }
    }

} // namespace

TEST(PlaBundleSolverInternalTest, AnalyticPointAndPoseJacobiansMatchCentralDifferences)
{
    CameraState camera = makeCamera(-1.5, 0.8);
    const double pose_delta[6] = {0.01, -0.02, 0.015, 0.1, -0.05, 0.02};
    camera.applyDeltaPose(pose_delta);
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    const auto observed = project(camera, point);
    const BAObservation observation{0, observed[0] + 0.4, observed[1] - 0.2, 0.75, 1.0};

    solver::ObservationLinearization linearization;
    ASSERT_TRUE(solver::linearizeObservation(
        camera, point, observation, plabundle::ImageRobustLoss::Huber, 3.0, &linearization));
    for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
    {
        for (int parameter = 0; parameter < 3; ++parameter)
        {
            EXPECT_NEAR(linearization.pointJacobian[static_cast<std::size_t>(pixel_axis * 3 + parameter)],
                        numericPointDerivative(camera, point, parameter, pixel_axis),
                        2.0e-4);
        }
        for (int parameter = 0; parameter < 6; ++parameter)
        {
            EXPECT_NEAR(linearization.cameraJacobian[static_cast<std::size_t>(pixel_axis * 6 + parameter)],
                        numericCameraDerivative(camera, point, parameter, pixel_axis),
                        2.0e-3);
        }
    }
}

TEST(PlaBundleSolverInternalTest, RollingShutterPoseJacobiansUseObservedLineEffectivePose)
{
    plabundle::FrameCamera frame = makeCamera(-1.5, 0.8).frameCamera();
    frame.projectionModel = plabundle::FrameProjectionModel::RollingShutter;
    frame.rollingShutter.referenceLinePixels = 384.0;
    frame.rollingShutter.secondsPerLine = 2.0e-4;
    frame.rollingShutter.linearVelocityWorldMetersPerSecond = {{2.0, -1.0, 0.5}};
    frame.rollingShutter.angularVelocityCameraRadiansPerSecond = {{0.2, -0.15, 0.1}};
    const CameraState camera(frame);
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    constexpr double observed_line = 520.0;
    const double world[3] = {point[0], point[1], point[2]};
    double predicted[2]{};
    double depth = 0.0;
    ASSERT_TRUE(camera.projectWorldPointWithDepthAtLine(world, observed_line, predicted, depth));
    const BAObservation observation{0, predicted[0] + 0.4, observed_line, 1.0, 1.0};

    solver::ObservationLinearization linearization;
    ASSERT_TRUE(solver::linearizeObservation(
        camera, point, observation, plabundle::ImageRobustLoss::Huber, 3.0, &linearization));
    for (int parameter = 0; parameter < 3; ++parameter)
    {
        constexpr double epsilon = 1.0e-6;
        auto plus_point = point;
        auto minus_point = point;
        plus_point[parameter] += epsilon;
        minus_point[parameter] -= epsilon;
        const double plus_world[3] = {plus_point[0], plus_point[1], plus_point[2]};
        const double minus_world[3] = {minus_point[0], minus_point[1], minus_point[2]};
        double plus[2]{};
        double minus[2]{};
        ASSERT_TRUE(camera.projectWorldPointWithDepthAtLine(plus_world, observed_line, plus, depth));
        ASSERT_TRUE(camera.projectWorldPointWithDepthAtLine(minus_world, observed_line, minus, depth));
        for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
        {
            const double numeric = (plus[pixel_axis] - minus[pixel_axis]) / (2.0 * epsilon);
            EXPECT_NEAR(linearization.pointJacobian[pixel_axis * 3 + parameter], numeric, 3.0e-4);
        }
    }
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        constexpr double epsilon = 1.0e-7;
        double delta[6]{};
        delta[parameter] = epsilon;
        CameraState plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        CameraState minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        double plus[2]{};
        double minus[2]{};
        ASSERT_TRUE(plus_camera.projectWorldPointWithDepthAtLine(world, observed_line, plus, depth));
        ASSERT_TRUE(minus_camera.projectWorldPointWithDepthAtLine(world, observed_line, minus, depth));
        for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
        {
            const double numeric = (plus[pixel_axis] - minus[pixel_axis]) / (2.0 * epsilon);
            EXPECT_NEAR(linearization.cameraJacobian[pixel_axis * 6 + parameter], numeric, 3.0e-3);
        }
    }
}

TEST(PlaBundleSolverInternalTest, SharedBrownJacobiansMatchCentralDifferences)
{
    const CameraState camera = makeCamera(-1.2, 0.7);
    CameraState reference = camera;
    reference.setIntrinsics(950.0, 920.0, 512.0, 384.0);
    const plabundle::CameraParameterBlock parameters{
        {940.0, std::log(0.985), 1.5, -2.0, -0.018, 0.0012, -0.00015, 0.00045, -0.00035}};
    const std::array<double, 3> point{{0.8, -0.6, 11.0}};
    const auto pixel = project(cameraWithSharedIntrinsics(camera, reference, parameters), point);
    const BAObservation observation{0, pixel[0] + 0.3, pixel[1] - 0.2, 1.0, 1.0};
    BAIntrinsicParameterMask active{};
    active.fill(true);

    solver::ObservationLinearization linearization;
    ASSERT_TRUE(solver::linearizeObservationWithSharedIntrinsics(camera,
                                                                 reference,
                                                                 parameters,
                                                                 active,
                                                                 point,
                                                                 observation,
                                                                 plabundle::ImageRobustLoss::Huber,
                                                                 3.0,
                                                                 &linearization));
    for (int parameter = 0; parameter < static_cast<int>(plabundle::kCameraParameterCount); ++parameter)
    {
        constexpr double epsilon = 1.0e-7;
        auto plus_parameters = parameters;
        auto minus_parameters = parameters;
        plus_parameters[static_cast<std::size_t>(parameter)] += epsilon;
        minus_parameters[static_cast<std::size_t>(parameter)] -= epsilon;
        const auto plus = project(cameraWithSharedIntrinsics(camera, reference, plus_parameters), point);
        const auto minus = project(cameraWithSharedIntrinsics(camera, reference, minus_parameters), point);
        for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
        {
            const double numeric =
                (plus[static_cast<std::size_t>(pixel_axis)] - minus[static_cast<std::size_t>(pixel_axis)]) /
                (2.0 * epsilon);
            EXPECT_NEAR(linearization.intrinsicJacobian[static_cast<std::size_t>(
                            pixel_axis * plabundle::kCameraParameterCount + parameter)],
                        numeric,
                        3.0e-4);
        }
    }
}

TEST(PlaBundleSolverInternalTest, SharedJacobiansMatchCentralDifferencesForEveryProjectionModel)
{
    const std::vector<plabundle::FrameProjectionModel> models = {
        plabundle::FrameProjectionModel::BrownConrady,
        plabundle::FrameProjectionModel::Fisheye,
        plabundle::FrameProjectionModel::Equidistant,
        plabundle::FrameProjectionModel::Equisolid,
        plabundle::FrameProjectionModel::Spherical,
        plabundle::FrameProjectionModel::Cylindrical,
        plabundle::FrameProjectionModel::RollingShutter,
    };
    const std::array<double, 3> point{{0.8, -0.6, 11.0}};
    for (const auto model : models)
    {
        SCOPED_TRACE(static_cast<int>(model));
        const CameraState camera = makeModelCamera(model);
        CameraState reference = camera;
        reference.setIntrinsics(950.0, 920.0, 512.0, 384.0);
        auto parameters = camera.parameterBlock();
        parameters[0] = 940.0;
        parameters[1] = std::log(0.985);
        parameters[2] = 1.5;
        parameters[3] = -2.0;
        const CameraState effective = cameraWithSharedIntrinsics(camera, reference, parameters);
        const auto pixel = project(effective, point);
        const BAObservation observation{0, pixel[0] + 0.3, pixel[1] - 0.2, 1.0, 1.0};
        const BAIntrinsicParameterMask active = camera.supportedParameters();

        solver::ObservationLinearization linearization;
        ASSERT_TRUE(solver::linearizeObservationWithSharedIntrinsics(camera,
                                                                     reference,
                                                                     parameters,
                                                                     active,
                                                                     point,
                                                                     observation,
                                                                     plabundle::ImageRobustLoss::Huber,
                                                                     3.0,
                                                                     &linearization));
        for (int parameter = 0; parameter < static_cast<int>(plabundle::kCameraParameterCount); ++parameter)
        {
            if (!active[static_cast<std::size_t>(parameter)])
            {
                EXPECT_DOUBLE_EQ(linearization.intrinsicJacobian[parameter], 0.0);
                EXPECT_DOUBLE_EQ(linearization.intrinsicJacobian[plabundle::kCameraParameterCount + parameter], 0.0);
                continue;
            }
            constexpr double epsilon = 1.0e-7;
            auto plus_parameters = parameters;
            auto minus_parameters = parameters;
            plus_parameters[static_cast<std::size_t>(parameter)] += epsilon;
            minus_parameters[static_cast<std::size_t>(parameter)] -= epsilon;
            const auto plus = project(cameraWithSharedIntrinsics(camera, reference, plus_parameters), point);
            const auto minus = project(cameraWithSharedIntrinsics(camera, reference, minus_parameters), point);
            for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
            {
                const double numeric =
                    (plus[static_cast<std::size_t>(pixel_axis)] - minus[static_cast<std::size_t>(pixel_axis)]) /
                    (2.0 * epsilon);
                EXPECT_NEAR(linearization.intrinsicJacobian[static_cast<std::size_t>(
                                pixel_axis * plabundle::kCameraParameterCount + parameter)],
                            numeric,
                            3.0e-4);
            }
        }
    }
}

TEST(PlaBundleSolverInternalTest, MeasurementScaleWhitensResidualAndNormalWeight)
{
    const CameraState camera = makeCamera(-1.5, 0.8);
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    const auto observed = project(camera, point);
    BAObservation fine{0, observed[0] + 0.4, observed[1] - 0.2, 0.75, 1.0};
    BAObservation coarse = fine;
    coarse.measurementScale = 2.0;

    solver::ObservationLinearization fine_linearization;
    solver::ObservationLinearization coarse_linearization;
    ASSERT_TRUE(solver::linearizeObservation(
        camera, point, fine, plabundle::ImageRobustLoss::LeastSquares, 1.0, &fine_linearization, true, false));
    ASSERT_TRUE(solver::linearizeObservation(
        camera, point, coarse, plabundle::ImageRobustLoss::LeastSquares, 1.0, &coarse_linearization, true, false));
    EXPECT_NEAR(coarse_linearization.normalWeight, fine_linearization.normalWeight / 4.0, 1.0e-14);
    EXPECT_NEAR(coarse_linearization.robustCost, fine_linearization.robustCost / 4.0, 1.0e-14);
}

TEST(PlaBundleSolverInternalTest, ImageRobustLossCostAndIrlsWeightMatchDefinitions)
{
    const auto least_squares = solver::evaluateImageRobustLoss(25.0, plabundle::ImageRobustLoss::LeastSquares, 2.0);
    EXPECT_DOUBLE_EQ(least_squares.cost, 12.5);
    EXPECT_DOUBLE_EQ(least_squares.weight, 1.0);

    const auto huber = solver::evaluateImageRobustLoss(25.0, plabundle::ImageRobustLoss::Huber, 2.0);
    EXPECT_DOUBLE_EQ(huber.cost, 8.0);
    EXPECT_DOUBLE_EQ(huber.weight, 0.4);

    const auto cauchy = solver::evaluateImageRobustLoss(25.0, plabundle::ImageRobustLoss::Cauchy, 2.0);
    EXPECT_NEAR(cauchy.cost, 2.0 * std::log(7.25), 1.0e-14);
    EXPECT_NEAR(cauchy.weight, 4.0 / 29.0, 1.0e-14);
}

TEST(PlaBundleSolverInternalTest, FixedAndSharedImageAssemblyUseConfiguredRobustLoss)
{
    const std::vector<CameraState> cameras{makeCamera(-1.5, 0.8), makeCamera(1.5, -0.8)};
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    BATrack track = makeTrack(cameras, point);
    track.observations[0].u += 3.0;
    track.observations[0].v += 4.0;
    const std::vector<BATrack> tracks{track};
    const auto expected = solver::evaluateImageRobustLoss(25.0, plabundle::ImageRobustLoss::Cauchy, 2.0);

    BAOptions fixed_options;
    fixed_options.refineCameraPose = false;
    fixed_options.imageRobustLoss = plabundle::ImageRobustLoss::Cauchy;
    fixed_options.imageRobustLossScalePixels = 2.0;
    const solver::ActiveProblem fixed_active = solver::prepareActiveProblem(cameras, tracks, fixed_options);
    const solver::OptimizationState fixed_state = solver::initializeState(cameras, tracks, fixed_options, fixed_active);
    solver::ObservationLinearization fixed;
    ASSERT_TRUE(solver::assembly_detail::linearizeImageObservation(
        cameras, fixed_options, fixed_active, fixed_state, 0, point, track.observations[0], 0, &fixed));
    EXPECT_NEAR(fixed.robustCost, expected.cost, 1.0e-12);
    EXPECT_NEAR(fixed.normalWeight, expected.weight, 1.0e-12);

    BAOptions shared_options = fixed_options;
    shared_options.refineSharedFocalLength = true;
    shared_options.cameraCalibrationGroupIds = {7, 7};
    const solver::ActiveProblem shared_active = solver::prepareActiveProblem(cameras, tracks, shared_options);
    const solver::OptimizationState shared_state =
        solver::initializeState(cameras, tracks, shared_options, shared_active);
    ASSERT_FALSE(shared_state.intrinsicGroups.empty());
    solver::ObservationLinearization shared;
    ASSERT_TRUE(solver::assembly_detail::linearizeImageObservation(
        cameras, shared_options, shared_active, shared_state, 0, point, track.observations[0], 0, &shared));
    EXPECT_NEAR(shared.robustCost, expected.cost, 1.0e-12);
    EXPECT_NEAR(shared.normalWeight, expected.weight, 1.0e-12);
}

TEST(PlaBundleSolverInternalTest, ActiveProblemMapsFixedGroupedAndPromotedBlocks)
{
    const std::vector<CameraState> cameras{makeCamera(-2.0, 0.0), makeCamera(0.0, 0.5), makeCamera(2.0, 0.0)};
    const std::vector<BATrack> tracks{makeTrack(cameras, {-0.4, 0.1, 8.0}), makeTrack(cameras, {0.5, -0.2, 9.0})};
    BAOptions options;
    options.fixedCameraIndices = {0};
    options.refineSharedFocalLength = true;
    options.cameraCalibrationGroupIds = {7, 7, 9};
    options.enableScaleBarConstraints = true;
    options.scaleBarConstraints.push_back({0, 1, 1.0, 0.1, 1.0, 0});

    const solver::ActiveProblem active = solver::prepareActiveProblem(cameras, tracks, options);
    EXPECT_EQ(active.activeTrackCount, 2);
    EXPECT_EQ(active.observationCount, 6);
    EXPECT_EQ(active.cameraBlockCount, 2);
    EXPECT_EQ(active.cameraBlock, (std::vector<int>{-1, 0, 1}));
    EXPECT_EQ(active.intrinsicBlockCount, 2);
    EXPECT_EQ(active.calibrationGroupByCamera, (std::vector<int>{0, 0, 1}));
    EXPECT_EQ(active.promotedTrackBlockCount, 2);
    EXPECT_EQ(active.trackBlockCount, 0);
    EXPECT_EQ(active.primaryBlockCount, 6);
}

TEST(PlaBundleSolverInternalTest, RigCaptureAndSensorJacobiansMatchCentralDifferences)
{
    const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    std::vector<plabundle::FrameCamera> models{makeCamera(0.0, 0.0).frameCamera(), makeCamera(0.0, 0.0).frameCamera()};
    plabundle::RigTopology rig;
    rig.captures = {{0, 0, identity, {-1.2, 0.4, 0.1}, false}};
    rig.sensors = {{0, 0, identity, {-0.35, 0.0, 0.0}, true}, {0, 1, identity, {0.45, 0.2, -0.05}, false}};
    rig.cameraBindings = {{0, 0, 0, 1}, {1, 0, 0, 0}};
    std::vector<plabundle::FrameCamera> frames;
    ASSERT_TRUE(plabundle::composeRigCameras(models, rig, &frames));
    const std::vector<CameraState> cameras = plabundle::internal::makeCameraStates(frames);
    const std::array<double, 3> point{{0.6, -0.3, 10.0}};
    const std::vector<BATrack> tracks{makeTrack(cameras, point)};
    BAOptions options;
    options.rig = rig;
    options.refineCameraPose = true;
    const solver::ActiveProblem active = solver::prepareActiveProblem(cameras, tracks, options);
    ASSERT_EQ(active.rigCaptureBlockCount, 1);
    ASSERT_EQ(active.rigSensorBlockCount, 1);
    const solver::OptimizationState state = solver::initializeState(cameras, tracks, options, active);
    solver::ObservationLinearization linearization;
    ASSERT_TRUE(solver::linearizeObservation(cameras[0],
                                             point,
                                             tracks[0].observations[0],
                                             plabundle::ImageRobustLoss::LeastSquares,
                                             1.0,
                                             &linearization,
                                             true,
                                             false));
    const auto terms = solver::assembly_detail::observationPrimaryTerms(options, active, state, 0, linearization);
    ASSERT_EQ(terms.count, 2U);

    for (std::size_t term_index = 0; term_index < terms.count; ++term_index)
    {
        const bool capture = terms.blocks[term_index] == active.rigCaptureBlockByCamera[0];
        const std::size_t topology_index = capture ? 0U : 1U;
        for (int parameter = 0; parameter < 6; ++parameter)
        {
            constexpr double epsilon = 1.0e-7;
            auto plus_rig = rig;
            auto minus_rig = rig;
            applyRigDelta(&plus_rig, capture, topology_index, parameter, epsilon);
            applyRigDelta(&minus_rig, capture, topology_index, parameter, -epsilon);
            std::vector<plabundle::FrameCamera> plus_frames;
            std::vector<plabundle::FrameCamera> minus_frames;
            ASSERT_TRUE(plabundle::composeRigCameras(models, plus_rig, &plus_frames));
            ASSERT_TRUE(plabundle::composeRigCameras(models, minus_rig, &minus_frames));
            plabundle::Projection plus;
            plabundle::Projection minus;
            ASSERT_TRUE(plabundle::projectWorldPoint(plus_frames[0], point, &plus));
            ASSERT_TRUE(plabundle::projectWorldPoint(minus_frames[0], point, &minus));
            for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
            {
                const double numeric = (plus.pixel[static_cast<std::size_t>(pixel_axis)] -
                                        minus.pixel[static_cast<std::size_t>(pixel_axis)]) /
                                       (2.0 * epsilon);
                EXPECT_NEAR(terms.jacobians[term_index][static_cast<std::size_t>(
                                pixel_axis * solver::kPrimaryBlockSize + parameter)],
                            numeric,
                            4.0e-3);
            }
        }
    }
}

TEST(PlaBundleSolverInternalTest, RigPosePriorJacobiansMatchCentralDifferences)
{
    const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    std::vector<plabundle::FrameCamera> models{makeCamera(0.0, 0.0).frameCamera(), makeCamera(0.0, 0.0).frameCamera()};
    plabundle::RigTopology rig;
    rig.captures = {{0, 0, identity, {-1.2, 0.4, 0.1}, false}};
    rig.sensors = {{0, 0, identity, {0.45, 0.2, -0.05}, false}, {0, 1, identity, {-0.35, 0.0, 0.0}, true}};
    rig.cameraBindings = {{0, 0, 0, 0}, {1, 0, 0, 1}};
    std::vector<plabundle::FrameCamera> frames;
    ASSERT_TRUE(plabundle::composeRigCameras(models, rig, &frames));
    const std::vector<CameraState> cameras = plabundle::internal::makeCameraStates(frames);
    const std::array<double, 3> point{{0.6, -0.3, 10.0}};
    const std::vector<BATrack> tracks{makeTrack(cameras, point)};
    BAOptions options;
    options.rig = rig;
    options.refineCameraPose = true;
    options.cameraPosePriorWeight = 3.0;
    const solver::ActiveProblem active = solver::prepareActiveProblem(cameras, tracks, options);
    ASSERT_EQ(active.rigCaptureBlockCount, 1);
    ASSERT_EQ(active.rigSensorBlockCount, 1);
    const solver::OptimizationState state = solver::initializeState(cameras, tracks, options, active);

    plabundle::internal::BACameraPosePrior prior;
    prior.enabled = true;
    prior.cameraToWorldRotation = frames[0].cameraToWorldRotation;
    prior.cameraCenter = {
        frames[0].cameraCenter[0] + 0.08, frames[0].cameraCenter[1] - 0.05, frames[0].cameraCenter[2] + 0.03};
    prior.uncertainty = plabundle::PosePriorUncertainty::Covariance;
    prior.tangentFrame = plabundle::PosePriorTangentFrame::PriorCamera;
    for (int axis = 0; axis < 3; ++axis)
    {
        prior.uncertaintyMatrix[static_cast<std::size_t>(axis * 6 + axis)] = 0.01;
        prior.uncertaintyMatrix[static_cast<std::size_t>((axis + 3) * 6 + axis + 3)] = 0.04;
    }
    prior.uncertaintyMatrix[3] = 0.002;
    prior.uncertaintyMatrix[18] = 0.002;

    solver::ConstraintLinearization linearization;
    ASSERT_TRUE(solver::linearizePosePrior(cameras[0], prior, options, &linearization));
    ASSERT_EQ(linearization.residualSize, 6);
    const auto terms = solver::assembly_detail::cameraPosePrimaryTerms(options,
                                                                       active,
                                                                       state,
                                                                       0,
                                                                       linearization.primaryJacobian.data(),
                                                                       linearization.residualSize,
                                                                       solver::kPrimaryBlockSize);
    ASSERT_EQ(terms.count, 2U);

    for (std::size_t term_index = 0; term_index < terms.count; ++term_index)
    {
        const bool capture = terms.blocks[term_index] == active.rigCaptureBlockByCamera[0];
        const std::size_t topology_index = 0;
        for (int parameter = 0; parameter < 6; ++parameter)
        {
            constexpr double epsilon = 1.0e-7;
            auto plus_rig = rig;
            auto minus_rig = rig;
            applyRigDelta(&plus_rig, capture, topology_index, parameter, epsilon);
            applyRigDelta(&minus_rig, capture, topology_index, parameter, -epsilon);
            std::vector<plabundle::FrameCamera> plus_frames;
            std::vector<plabundle::FrameCamera> minus_frames;
            ASSERT_TRUE(plabundle::composeRigCameras(models, plus_rig, &plus_frames));
            ASSERT_TRUE(plabundle::composeRigCameras(models, minus_rig, &minus_frames));
            solver::ConstraintLinearization plus;
            solver::ConstraintLinearization minus;
            ASSERT_TRUE(solver::linearizePosePrior(CameraState(plus_frames[0]), prior, options, &plus));
            ASSERT_TRUE(solver::linearizePosePrior(CameraState(minus_frames[0]), prior, options, &minus));
            for (int residual = 0; residual < linearization.residualSize; ++residual)
            {
                const double numeric = (plus.residual[static_cast<std::size_t>(residual)] -
                                        minus.residual[static_cast<std::size_t>(residual)]) /
                                       (2.0 * epsilon);
                EXPECT_NEAR(terms.jacobians[term_index]
                                           [static_cast<std::size_t>(residual * solver::kPrimaryBlockSize + parameter)],
                            numeric,
                            2.0e-5);
            }
        }
    }
}

TEST(PlaBundleSolverInternalTest, InvalidAcceptedStateProjectionRemainsAHardError)
{
    const std::vector<CameraState> cameras{makeCamera(-1.0, 0.0), makeCamera(1.0, 0.0)};
    const std::vector<BATrack> tracks{makeTrack(cameras, {0.0, 0.0, 4.0})};
    BAOptions options;
    options.refineCameraPose = false;
    const solver::ActiveProblem active = solver::prepareActiveProblem(cameras, tracks, options);
    ASSERT_EQ(active.activeTrackCount, 1);
    solver::OptimizationState state = solver::initializeState(cameras, tracks, options, active);
    state.points[0][2] = -1.0;

    EXPECT_THROW(solver::evaluateObjective(cameras, tracks, options, active, state, 0), solver::InvalidProjectionError);
}

TEST(PlaBundleSolverInternalTest, ParallelWorkerRethrowsFirstException)
{
    EXPECT_THROW(plabundle::internal::parallelForIndices(64,
                                                         4,
                                                         [](std::size_t index)
                                                         {
                                                             if (index == 3)
                                                             {
                                                                 throw std::runtime_error("worker failure");
                                                             }
                                                         }),
                 std::runtime_error);
}
