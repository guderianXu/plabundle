#include <plabundle/options.h>
#include <plabundle/problem.h>
#include <plabundle/solver.h>

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <stop_token>
#include <string>

namespace
{

    plabundle::FrameCamera makeCamera(double center_x)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter[0] = center_x;
        camera.focalXPixels = 1000.0;
        camera.focalYPixels = 1000.0;
        camera.principalXPixel = 500.0;
        camera.principalYPixel = 400.0;
        return camera;
    }

    plabundle::Problem makeProblem()
    {
        plabundle::Problem problem;
        problem.cameras = {makeCamera(0.0), makeCamera(1.0)};
        plabundle::Track track;
        track.initialPoint = {0.0, 0.0, 5.0};
        track.observations = {{0, 500.0, 400.0, 1.0, 1.0}, {1, 300.0, 400.0, 1.0, 1.0}};
        problem.tracks.push_back(track);
        return problem;
    }

} // namespace

TEST(PlaBundleProblemTest, ValidatesAndSummarizesUsableTracks)
{
    const plabundle::Problem problem = makeProblem();
    std::string error;
    EXPECT_TRUE(plabundle::validateProblem(problem, &error)) << error;
    const plabundle::ProblemStats stats = plabundle::summarizeProblem(problem);
    EXPECT_EQ(stats.cameraCount, 2);
    EXPECT_EQ(stats.trackCount, 1);
    EXPECT_EQ(stats.observationCount, 2);
}

TEST(PlaBundleProblemTest, RejectsDuplicateFixedIndicesAndSingleViewTracks)
{
    plabundle::Problem problem = makeProblem();
    problem.fixedCameraIndices = {0, 0};
    EXPECT_FALSE(plabundle::validateProblem(problem));

    problem = makeProblem();
    problem.tracks[0].observations[1].cameraIndex = 0;
    EXPECT_FALSE(plabundle::validateProblem(problem));
}

TEST(PlaBundleProblemTest, RejectsIncompleteGaugeAndUnderdeterminedLaserPoints)
{
    plabundle::Problem problem = makeProblem();
    problem.gauge.referenceAnchorCameraIndex = 0;
    EXPECT_FALSE(plabundle::validateProblem(problem));

    problem = makeProblem();
    problem.gauge.policy = static_cast<plabundle::GaugePolicy>(99);
    EXPECT_FALSE(plabundle::validateProblem(problem));

    problem = makeProblem();
    problem.gauge.referenceBaseline = -1.0;
    EXPECT_FALSE(plabundle::validateProblem(problem));

    problem = makeProblem();
    plabundle::LaserRangeConstraint constrained;
    constrained.cameraIndex = 0;
    constrained.initialPoint = {0.0, 0.0, 5.0};
    constrained.observedRangeMeters = 5.0;
    constrained.pointMode = plabundle::LaserPointMode::Constrained;
    problem.laserRangeConstraints.push_back(constrained);
    EXPECT_FALSE(plabundle::validateProblem(problem));

    problem = makeProblem();
    constrained.pointMode = static_cast<plabundle::LaserPointMode>(99);
    problem.laserRangeConstraints.push_back(constrained);
    EXPECT_FALSE(plabundle::validateProblem(problem));

    problem = makeProblem();
    plabundle::LaserRangeConstraint free_point;
    free_point.cameraIndex = 0;
    free_point.initialPoint = {0.0, 0.0, 5.0};
    free_point.observedRangeMeters = 5.0;
    free_point.pointMode = plabundle::LaserPointMode::Free;
    free_point.measuredImageObservations = {{0, 500.0, 400.0, 1.0, 1.0}};
    problem.laserRangeConstraints.push_back(free_point);
    EXPECT_FALSE(plabundle::validateProblem(problem));
}

TEST(PlaBundleProblemTest, SolverRunsPlaMatrixCpuAndPublishesAUsableResult)
{
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    const plabundle::Result result = plabundle::Solver().solve(makeProblem(), options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::Success);
    EXPECT_EQ(result.requestedBackend, plabundle::Backend::Auto);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_TRUE(result.usable());
    EXPECT_FALSE(result.usedGpu);
    ASSERT_EQ(result.points.size(), 1U);
    EXPECT_TRUE(result.points.front().valid);
    EXPECT_NEAR(result.points.front().point[0], 0.0, 1.0e-10);
    EXPECT_NEAR(result.points.front().point[1], 0.0, 1.0e-10);
    EXPECT_NEAR(result.points.front().point[2], 5.0, 1.0e-10);
    EXPECT_NEAR(result.quality.meanRmsAfter, 0.0, 1.0e-10);
    EXPECT_NE(result.plaMatrix.linearSolverName, "none");
    EXPECT_TRUE(plabundle::Solver::isBackendAvailable(plabundle::Backend::Auto));
    EXPECT_TRUE(plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCpu));
    EXPECT_FALSE(plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda, 9999));
    EXPECT_TRUE(plabundle::Solver::backendCapabilities(plabundle::Backend::PlaMatrixCpu).refinesCameraPose);
}

TEST(PlaBundleProblemTest, AcceleratedBackendsRejectInvalidDeviceWithoutFallback)
{
    for (const plabundle::Backend backend : {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl})
    {
        plabundle::Options options;
        options.backend = backend;
        options.plaMatrixDevice = 9999;
        options.allowBackendFallback = false;
        const plabundle::Result result = plabundle::Solver().solve(makeProblem(), options);
        EXPECT_EQ(result.status, plabundle::SolveStatus::BackendUnavailable);
        EXPECT_EQ(result.requestedBackend, backend);
        EXPECT_EQ(result.usedBackend, backend);
        EXPECT_FALSE(result.usable());
        EXPECT_FALSE(result.usedGpu);
        EXPECT_FALSE(result.backendFallback);
        EXPECT_FALSE(result.backendMessage.empty());
    }
}

TEST(PlaBundleProblemTest, AcceleratedBackendsFallbackToCpuForInvalidDevice)
{
    for (const plabundle::Backend backend : {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl})
    {
        plabundle::Options options;
        options.backend = backend;
        options.plaMatrixDevice = 9999;
        options.allowBackendFallback = true;
        const plabundle::Result result = plabundle::Solver().solve(makeProblem(), options);
        ASSERT_TRUE(result.usable()) << result.backendMessage;
        EXPECT_EQ(result.requestedBackend, backend);
        EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
        EXPECT_TRUE(result.backendFallback);
        EXPECT_FALSE(result.usedGpu);
        EXPECT_NE(result.backendSelectionReason.find("fallback_to_plamatrix_cpu"), std::string::npos);
    }
}

TEST(PlaBundleProblemTest, SolverHonorsPreCancelledAtomicFlag)
{
    plabundle::Options options;
    options.cancelFlag = std::make_shared<std::atomic<bool>>(true);
    const plabundle::Result result = plabundle::Solver().solve(makeProblem(), options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::Cancelled);
    EXPECT_FALSE(result.usable());
}

TEST(PlaBundleProblemTest, SolverHonorsPreCancelledStopToken)
{
    std::stop_source stop_source;
    ASSERT_TRUE(stop_source.request_stop());
    plabundle::Options options;
    options.stopToken = stop_source.get_token();
    const plabundle::Result result = plabundle::Solver().solve(makeProblem(), options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::Cancelled);
    EXPECT_FALSE(result.usable());
}

TEST(PlaBundleProblemTest, SolverRejectsInvalidOptionsBeforeEngineDispatch)
{
    plabundle::Options options;
    options.plaMatrixPreconditionerClusterSize = 0;
    const plabundle::Result result = plabundle::Solver().solve(makeProblem(), options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::InvalidInput);
    EXPECT_FALSE(result.backendMessage.empty());
}

TEST(PlaBundleProblemTest, PublicOptionValidationMatchesStrictCpuKernelBounds)
{
    plabundle::Options options;
    options.maxSharedPrincipalPointOffsetFraction = 0.0;
    EXPECT_FALSE(plabundle::validateOptions(options));

    options = plabundle::Options();
    options.maxSharedRadialK1Abs = 0.0;
    EXPECT_FALSE(plabundle::validateOptions(options));

    options = plabundle::Options();
    options.sharedLowOrderDistortionScale = 0.5;
    EXPECT_FALSE(plabundle::validateOptions(options));
}

TEST(PlaBundleProblemTest, IntrinsicMaskRefinesOnlyEnabledParameters)
{
    plabundle::Options options;
    options.refineSharedRadialDistortion = true;
    options.useSharedIntrinsicParameterMask = true;
    options.sharedIntrinsicParameterMask.fill(false);
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::RadialK1)] = true;
    EXPECT_TRUE(plabundle::sharedIntrinsicParameterEnabled(options, plabundle::IntrinsicParameter::RadialK1));
    EXPECT_FALSE(plabundle::sharedIntrinsicParameterEnabled(options, plabundle::IntrinsicParameter::RadialK2));
}
