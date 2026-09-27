#include <plabundle/linescan.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>

namespace
{

    class LinearTrajectory final : public plabundle::linescan::PoseTrajectory
    {
    public:
        LinearTrajectory(std::array<double, 3> origin, std::array<double, 3> velocity)
            : _origin(origin), _velocity(velocity)
        {
        }

        bool evaluate(double timeSeconds, plabundle::linescan::SensorPose* pose) const override
        {
            if (!pose || !std::isfinite(timeSeconds))
            {
                return false;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                pose->centerMeters[axis] = _origin[axis] + _velocity[axis] * timeSeconds;
            }
            pose->worldToSensorRotation = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
            return true;
        }

    private:
        std::array<double, 3> _origin;
        std::array<double, 3> _velocity;
    };

    plabundle::linescan::Problem makeProblem()
    {
        plabundle::linescan::Problem problem;
        problem.cameraParameters.resize(2);
        problem.tiePoints = {{{1.0, 2.0, 10.0}}};
        problem.imageObservations = {{0, 0, 1.0, 2.0}, {1, 0, 1.0, 2.0}};
        problem.projection = [](const plabundle::linescan::ImageObservation& observation,
                                const std::array<double, 6>& camera,
                                const std::array<double, 3>& point,
                                std::array<double, 2>* residual)
        {
            (*residual)[0] = point[0] - camera[0] - observation.samplePixels;
            (*residual)[1] = point[1] - camera[1] - observation.linePixels;
            return true;
        };
        problem.laserPoints.push_back({{{0.0, 0.0, 10.0}}, {{0.0, 0.0, 10.0}}});
        problem.laserObservations.push_back({0, 0, {{0.0, 0.0, 0.0}}, 12.0, 0.1});
        return problem;
    }

    plabundle::linescan::CameraModel makeCameraModel(const std::array<double, 3>& center)
    {
        plabundle::linescan::CameraModel model;
        model.timing.referenceLinePixels = 10.0;
        model.timing.referenceTimeSeconds = 2.0;
        model.timing.secondsPerLine = 0.01;
        model.trajectory = std::make_shared<LinearTrajectory>(center, std::array<double, 3>{});
        model.focalSamplePixels = 100.0;
        model.focalLinePixels = 100.0;
        return model;
    }

    plabundle::linescan::Problem makeFormalProblem()
    {
        plabundle::linescan::Problem problem;
        problem.cameraParameters.resize(2);
        problem.cameraModels = {makeCameraModel({{0.0, 0.0, 0.0}}), makeCameraModel({{1.0, 0.0, 0.0}})};
        problem.tiePoints = {{{0.0, 0.0, 10.0}}};
        problem.imageObservations = {{0, 0, 0.0, 10.0}, {1, 0, -10.0, 10.0}};
        return problem;
    }

} // namespace

TEST(LineScanTest, FormalLineTimingAndAnalyticJacobiansMatchCentralDifferences)
{
    auto model = makeCameraModel({{2.0, -1.0, 0.5}});
    model.trajectory = std::make_shared<LinearTrajectory>(std::array<double, 3>{{2.0, -1.0, 0.5}},
                                                          std::array<double, 3>{{0.2, -0.1, 0.05}});
    const plabundle::linescan::ImageObservation observation{0, 0, 16.0, 25.0};
    const std::array<double, 6> camera{{0.1, -0.2, 0.3, 0.02, -0.03, 0.04}};
    const std::array<double, 3> point{{4.0, 1.5, 12.0}};
    plabundle::linescan::ProjectionState state;
    ASSERT_TRUE(plabundle::linescan::prepareProjectionState(model, observation.linePixels, &state));
    EXPECT_NEAR(state.acquisitionTimeSeconds, 2.15, 1.0e-14);

    plabundle::linescan::ProjectionEvaluation analytic;
    ASSERT_TRUE(plabundle::linescan::evaluateProjection(model, state, observation, camera, point, &analytic));
    for (int block = 0; block < 2; ++block)
    {
        const int columns = block == 0 ? 6 : 3;
        for (int column = 0; column < columns; ++column)
        {
            const double step = block == 0 && column >= 3 ? 1.0e-7 : 1.0e-6;
            auto plus_camera = camera;
            auto minus_camera = camera;
            auto plus_point = point;
            auto minus_point = point;
            if (block == 0)
            {
                plus_camera[column] += step;
                minus_camera[column] -= step;
            }
            else
            {
                plus_point[column] += step;
                minus_point[column] -= step;
            }
            plabundle::linescan::ProjectionEvaluation plus;
            plabundle::linescan::ProjectionEvaluation minus;
            ASSERT_TRUE(
                plabundle::linescan::evaluateProjection(model, state, observation, plus_camera, plus_point, &plus));
            ASSERT_TRUE(
                plabundle::linescan::evaluateProjection(model, state, observation, minus_camera, minus_point, &minus));
            for (int row = 0; row < 2; ++row)
            {
                const double numeric = (plus.residualPixels[row] - minus.residualPixels[row]) / (2.0 * step);
                const double actual =
                    block == 0 ? analytic.cameraJacobian[row * 6 + column] : analytic.pointJacobian[row * 3 + column];
                EXPECT_NEAR(actual, numeric, 2.0e-5);
            }
        }
    }
}

TEST(LineScanTest, FormalCameraModelSolvesWithoutCompatibilityProjection)
{
    const auto problem = makeFormalProblem();
    plabundle::linescan::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    const auto result = plabundle::linescan::solve(problem, options);
    EXPECT_TRUE(result.usable()) << result.message;
    EXPECT_LT(result.refinedImageRmsPixels, 1.0e-12);
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(LineScanTest, ReportsUnobservableCameraAndSingleViewPoint)
{
    auto unobserved_camera = makeProblem();
    unobserved_camera.cameraParameters.emplace_back();
    const auto camera_result = plabundle::linescan::solve(unobserved_camera);
    ASSERT_FALSE(camera_result.usable());
    ASSERT_EQ(camera_result.diagnostics.size(), 1U);
    EXPECT_EQ(camera_result.diagnostics[0].code, plabundle::linescan::DiagnosticCode::UnobservedCamera);
    EXPECT_EQ(camera_result.diagnostics[0].cameraIndex, 2);

    auto single_view = makeProblem();
    single_view.tiePoints.push_back({{2.0, 2.0, 10.0}});
    single_view.imageObservations.push_back({0, 1, 2.0, 2.0});
    const auto point_result = plabundle::linescan::solve(single_view);
    ASSERT_FALSE(point_result.usable());
    ASSERT_EQ(point_result.diagnostics.size(), 1U);
    EXPECT_EQ(point_result.diagnostics[0].code, plabundle::linescan::DiagnosticCode::SingleViewPoint);
    EXPECT_EQ(point_result.diagnostics[0].pointIndex, 1);
}

TEST(LineScanTest, ReportsInvalidTimingAndDegenerateProjection)
{
    auto invalid_timing = makeFormalProblem();
    invalid_timing.cameraModels[0].timing.secondsPerLine = 0.0;
    const auto timing_result = plabundle::linescan::solve(invalid_timing);
    ASSERT_FALSE(timing_result.usable());
    ASSERT_EQ(timing_result.diagnostics.size(), 1U);
    EXPECT_EQ(timing_result.diagnostics[0].code, plabundle::linescan::DiagnosticCode::InvalidCameraModel);
    EXPECT_EQ(timing_result.diagnostics[0].cameraIndex, 0);

    auto invalid_depth = makeFormalProblem();
    invalid_depth.tiePoints[0][2] = -10.0;
    const auto depth_result = plabundle::linescan::solve(invalid_depth);
    ASSERT_FALSE(depth_result.usable());
    ASSERT_EQ(depth_result.diagnostics.size(), 1U);
    EXPECT_EQ(depth_result.diagnostics[0].code, plabundle::linescan::DiagnosticCode::DegenerateProjection);
}

TEST(LineScanTest, FixedRangeImprovesWithoutDegradingImages)
{
    const auto problem = makeProblem();
    plabundle::linescan::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.enableLaserRangeConstraints = true;
    options.maximumIterations = 30;
    const plabundle::linescan::Result result = plabundle::linescan::solve(problem, options);
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.solutionUsable);
    EXPECT_TRUE(result.usable());
    EXPECT_LT(result.refinedLaserRangeRmsMeters, result.initialLaserRangeRmsMeters);
    EXPECT_LT(result.refinedImageRmsPixels, 1.0e-6);
    ASSERT_EQ(result.refinedCameraParameters.size(), problem.cameraParameters.size());
    ASSERT_EQ(result.refinedTiePoints.size(), problem.tiePoints.size());
    ASSERT_EQ(result.refinedLaserPoints.size(), problem.laserPoints.size());
    EXPECT_LT(result.refinedCameraParameters[0][2], -1.0);
    EXPECT_EQ(problem.cameraParameters[0][2], 0.0);
    EXPECT_EQ(problem.tiePoints[0], (std::array<double, 3>{{1.0, 2.0, 10.0}}));
    EXPECT_EQ(problem.laserPoints[0].refinedMeters, problem.laserPoints[0].initialMeters);
}

TEST(LineScanTest, DisabledRangeDoesNotActivateConstrainedPointBlock)
{
    auto problem = makeProblem();
    problem.laserPoints[0].mode = plabundle::linescan::LaserPointMode::Constrained;
    plabundle::linescan::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.enableLaserRangeConstraints = false;
    plabundle::linescan::Result result;
    std::string error;
    ASSERT_TRUE(plabundle::linescan::solve(&problem, options, &result, &error)) << error;
    EXPECT_LT(result.refinedImageRmsPixels, 1.0e-12);
    EXPECT_EQ(problem.laserPoints[0].refinedMeters, problem.laserPoints[0].initialMeters);
}

TEST(LineScanTest, RejectsInvalidIndicesWithoutChangingInput)
{
    auto problem = makeProblem();
    problem.imageObservations[0].cameraIndex = 7;
    const auto original = problem.cameraParameters;
    const plabundle::linescan::Result result = plabundle::linescan::solve(problem);
    EXPECT_FALSE(result.usable());
    EXPECT_EQ(result.terminationType, "INVALID_INPUT");
    EXPECT_FALSE(result.message.empty());
    EXPECT_TRUE(result.refinedCameraParameters.empty());
    EXPECT_TRUE(result.refinedTiePoints.empty());
    EXPECT_TRUE(result.refinedLaserPoints.empty());
    EXPECT_EQ(problem.cameraParameters, original);
}

TEST(LineScanTest, CancellationDoesNotPublishPartialSolution)
{
    const auto problem = makeProblem();
    const auto original = problem.cameraParameters;
    plabundle::linescan::Options options;
    options.cancelFlag = std::make_shared<std::atomic<bool>>(true);
    const plabundle::linescan::Result result = plabundle::linescan::solve(problem, options);
    EXPECT_FALSE(result.usable());
    EXPECT_EQ(result.terminationType, "CANCELLED");
    EXPECT_TRUE(result.refinedCameraParameters.empty());
    EXPECT_TRUE(result.refinedTiePoints.empty());
    EXPECT_TRUE(result.refinedLaserPoints.empty());
    EXPECT_EQ(problem.cameraParameters, original);
}

TEST(LineScanTest, CompatibilityEntryPointPublishesAcceptedSolution)
{
    auto problem = makeProblem();
    plabundle::linescan::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.enableLaserRangeConstraints = true;
    options.maximumIterations = 30;
    plabundle::linescan::Result result;
    std::string error;

    ASSERT_TRUE(plabundle::linescan::solve(&problem, options, &result, &error)) << error;
    ASSERT_TRUE(result.usable());
    EXPECT_EQ(problem.cameraParameters, result.refinedCameraParameters);
    EXPECT_EQ(problem.tiePoints, result.refinedTiePoints);
    ASSERT_EQ(problem.laserPoints.size(), result.refinedLaserPoints.size());
    EXPECT_EQ(problem.laserPoints[0].refinedMeters, result.refinedLaserPoints[0]);
}

TEST(LineScanTest, CompatibilityEntryPointDoesNotPublishRejectedSolution)
{
    auto problem = makeProblem();
    problem.imageObservations[0].cameraIndex = 7;
    const auto original_cameras = problem.cameraParameters;
    const auto original_points = problem.tiePoints;
    const auto original_laser_points = problem.laserPoints;
    plabundle::linescan::Result result;
    std::string error;

    EXPECT_FALSE(plabundle::linescan::solve(&problem, {}, &result, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(result.terminationType, "INVALID_INPUT");
    EXPECT_EQ(problem.cameraParameters, original_cameras);
    EXPECT_EQ(problem.tiePoints, original_points);
    ASSERT_EQ(problem.laserPoints.size(), original_laser_points.size());
    EXPECT_EQ(problem.laserPoints[0].refinedMeters, original_laser_points[0].refinedMeters);
}
