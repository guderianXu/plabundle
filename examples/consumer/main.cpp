#include <plabundle/solver.h>
#include <plabundle/version.h>

#include <iostream>

int main()
{
    plabundle::FrameCamera left;
    left.cameraCenter = {-1.0, 0.0, 0.0};
    left.focalXPixels = 1000.0;
    left.focalYPixels = 1000.0;
    left.principalXPixel = 500.0;
    left.principalYPixel = 400.0;
    plabundle::FrameCamera right = left;
    right.cameraCenter = {1.0, 0.0, 0.0};

    plabundle::Track track;
    track.initialPoint = {0.2, -0.1, 6.0};
    track.observations = {{0, 700.0, 400.0, 1.0, 1.0}, {1, 300.0, 400.0, 1.0, 1.0}};

    plabundle::Problem problem;
    problem.cameras = {left, right};
    problem.tracks = {track};
    plabundle::Options options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
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
