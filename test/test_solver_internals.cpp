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
                                           const std::array<double, 9>& parameters)
    {
        CameraState camera = poseCamera;
        camera.setIntrinsics(parameters[0],
                             parameters[0] * std::exp(parameters[1]),
                             referenceCamera.principalX() + parameters[2],
                             referenceCamera.principalY() + parameters[3]);
        camera.setDistortion(parameters[4], parameters[5], parameters[6], parameters[7], parameters[8]);
        return camera;
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
    ASSERT_TRUE(solver::linearizeObservation(camera, point, observation, 3.0, &linearization));
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

TEST(PlaBundleSolverInternalTest, SharedBrownJacobiansMatchCentralDifferences)
{
    const CameraState camera = makeCamera(-1.2, 0.7);
    CameraState reference = camera;
    reference.setIntrinsics(950.0, 920.0, 512.0, 384.0);
    const std::array<double, 9> parameters{
        {940.0, std::log(0.985), 1.5, -2.0, -0.018, 0.0012, -0.00015, 0.00045, -0.00035}};
    const std::array<double, 3> point{{0.8, -0.6, 11.0}};
    const auto pixel = project(cameraWithSharedIntrinsics(camera, reference, parameters), point);
    const BAObservation observation{0, pixel[0] + 0.3, pixel[1] - 0.2, 1.0, 1.0};
    BAIntrinsicParameterMask active{};
    active.fill(true);

    solver::ObservationLinearization linearization;
    ASSERT_TRUE(solver::linearizeObservationWithSharedIntrinsics(
        camera, reference, parameters, active, point, observation, 3.0, &linearization));
    for (int parameter = 0; parameter < 9; ++parameter)
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
            EXPECT_NEAR(
                linearization.intrinsicJacobian[static_cast<std::size_t>(pixel_axis * 9 + parameter)], numeric, 3.0e-4);
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
    ASSERT_TRUE(solver::linearizeObservation(camera, point, fine, 0.0, &fine_linearization, true, false));
    ASSERT_TRUE(solver::linearizeObservation(camera, point, coarse, 0.0, &coarse_linearization, true, false));
    EXPECT_NEAR(coarse_linearization.normalWeight, fine_linearization.normalWeight / 4.0, 1.0e-14);
    EXPECT_NEAR(coarse_linearization.robustCost, fine_linearization.robustCost / 4.0, 1.0e-14);
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
