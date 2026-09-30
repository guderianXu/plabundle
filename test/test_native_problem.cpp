#include <plabundle/problem.h>

#include <placamera/frame_camera.h>

#include <gtest/gtest.h>

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
