#include <plabundle/solver.h>
#include <plabundle/version.h>
#include <placamera/frame_numeric_state.h>

#include <iostream>
#include <string>
#include <utility>

int main()
{
    const auto makeCamera = [](std::string instanceId, std::string imageId, double centerX)
    {
        placamera::FrameIntrinsics intrinsics;
        intrinsics.focalX = 1000.0;
        intrinsics.focalY = 1000.0;
        intrinsics.principalX = 500.0;
        intrinsics.principalY = 400.0;
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("consumer-definition"),
            intrinsics,
            {},
            placamera::PixelConvention::PixelCenter,
            placamera::FrameId("world"));
        const auto model = placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(std::move(instanceId)),
            placamera::ImageId(std::move(imageId)),
            definition,
            {1000, 800},
            placamera::Pose::create(
                placamera::FrameId("world"),
                {centerX, 0.0, 0.0},
                {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        return placamera::FramePinholeNumericState::fromModel(model);
    };

    const auto left = makeCamera("left", "left-image", -1.0);
    auto right = makeCamera("right", "right-image", 1.0);
    right.setPose(placamera::Pose::create(
        placamera::FrameId("world"),
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

    plabundle::Track track;
    track.initialPoint = {0.2, -0.1, 6.0};
    track.observations = {{0, 700.0, 400.0, 1.0, 1.0}, {1, 300.0, 400.0, 1.0, 1.0}};

    plabundle::Problem problem;
    problem.cameras = {left, right};
    problem.tracks = {track};
    plabundle::SolveOptions options;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    if (!result.usable())
    {
        std::cerr << result.backendMessage << '\n';
        return 1;
    }

    std::cout << "PlaBundle " << plabundle::kVersionString << ": " << result.quality.optimizedTracks
              << " optimized track, RMS=" << result.quality.meanRmsAfter << '\n';
    return 0;
}
