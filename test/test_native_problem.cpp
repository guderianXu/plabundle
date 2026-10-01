#include <plabundle/problem.h>
#include <plabundle/solver.h>

#include <placamera/frame_camera.h>

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

namespace
{
    placamera::FramePinholeNumericState makeCamera(const char* instance,
                                                    const char* image,
                                                    double center_x)
    {
        placamera::FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 100.0;
        intrinsics.principalX = 50.0;
        intrinsics.principalY = 50.0;
        const placamera::FrameId world("world");
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("definition"), intrinsics, {}, placamera::PixelConvention::PixelCenter, world);
        const auto model = placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(instance),
            placamera::ImageId(image),
            definition,
            placamera::ImageSize{100, 100},
            placamera::Pose::create(world,
                                    {center_x, 0.0, 0.0},
                                    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        return placamera::FramePinholeNumericState::fromModel(model);
    }
}

TEST(PlaBundleNativeProblemTest, AcceptsPlaCameraNumericStates)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera("camera-0", "image-0", 0.0), makeCamera("camera-1", "image-1", 1.0)};
    problem.tracks.push_back({{0.0, 0.0, 5.0}, {{0, 50.0, 50.0, 1.0, 1.0}, {1, 30.0, 50.0, 1.0, 1.0}}});

    std::string error;
    EXPECT_TRUE(plabundle::validateProblem(problem, &error)) << error;
    const auto stats = plabundle::summarizeProblem(problem);
    EXPECT_EQ(stats.cameraCount, 2);
    EXPECT_EQ(stats.observationCount, 2);
}

TEST(PlaBundleNativeProblemTest, ReferenceOnlineSchurSolvesUsingPlaMatrixSmallInverse)
{
    plabundle::Problem problem;
    const std::vector<placamera::FramePinholeNumericState> truth_cameras{makeCamera("camera-0", "image-0", 0.0),
                                                                         makeCamera("camera-1", "image-1", 1.0),
                                                                         makeCamera("camera-2", "image-2", 2.0)};
    problem.cameras = truth_cameras;
    problem.cameras[2].applyPoseDelta({0.001, -0.001, 0.0005, 0.02, -0.01, 0.005});
    problem.fixedCameraIndices = {0, 1};
    problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;

    for (int index = 0; index < 24; ++index)
    {
        const std::array<double, 3> truth{{0.5 * static_cast<double>(index % 4) - 0.75,
                                           0.3 * static_cast<double>((index / 4) % 3) - 0.3,
                                           5.0 + 0.2 * static_cast<double>(index % 5)}};
        plabundle::Track track;
        track.initialPoint = {{truth[0] + 0.03, truth[1] - 0.02, truth[2] + 0.04}};
        for (int camera_index = 0; camera_index < 3; ++camera_index)
        {
            const auto& camera = truth_cameras[static_cast<std::size_t>(camera_index)];
            const placamera::GroundCoordinate ground{camera.groundFrame(), truth};
            const auto projection = camera.groundToImage(ground);
            ASSERT_TRUE(projection);
            track.observations.push_back(
                {camera_index, projection.value().image.sample, projection.value().image.line, 1.0, 1.0});
        }
        problem.tracks.push_back(std::move(track));
    }

    plabundle::SolveOptions options;
    options.backend.requested = plabundle::Backend::PlaMatrixCpu;
    options.backend.allowFallback = false;
    options.solver.useReferenceOnlineSchur = true;
    options.solver.maxIterations = 8;
    options.solver.numThreads = 1;
    options.solver.enablePointFilter = false;
    options.solver.logIterationProgress = false;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    EXPECT_TRUE(result.plaMatrix.referenceOnlineSchurUsed) << result.backendMessage;
    EXPECT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_LT(result.plaMatrix.finalCost, result.plaMatrix.initialCost);
}
