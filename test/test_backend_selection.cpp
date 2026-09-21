#include <plabundle/solver.h>

#include <gtest/gtest.h>

#include <array>

namespace
{
    plabundle::FrameCamera makeCamera(double centerX)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter = {centerX, 0.0, 0.0};
        camera.focalXPixels = 1000.0;
        camera.focalYPixels = 1000.0;
        camera.principalXPixel = 500.0;
        camera.principalYPixel = 400.0;
        return camera;
    }

    plabundle::Problem makeProblem()
    {
        plabundle::Problem problem;
        problem.cameras = {makeCamera(-1.0), makeCamera(1.0)};
        plabundle::Track track;
        track.initialPoint = {0.0, 0.0, 5.0};
        track.observations = {{0, 700.0, 400.0, 1.0, 1.0}, {1, 300.0, 400.0, 1.0, 1.0}};
        problem.tracks.push_back(track);
        return problem;
    }

} // namespace

TEST(PlaBundleBackendSelectionTest, ExplicitBackendBypassesAutoScalePolicy)
{
    const plabundle::Problem problem = makeProblem();
    for (const plabundle::Backend backend :
         {plabundle::Backend::PlaMatrixCpu, plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl})
    {
        plabundle::Options options;
        options.backend = backend;
        const plabundle::BackendDecision decision = plabundle::Solver::decideBackendForProblem(problem, options);
        EXPECT_EQ(decision.backend, backend);
        EXPECT_EQ(decision.reason, "explicit_backend");
    }
}

TEST(PlaBundleBackendSelectionTest, PointOnlyAndSmallJointProblemsUseCpu)
{
    const plabundle::Problem problem = makeProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    options.refineCameraPose = false;
    plabundle::BackendDecision decision = plabundle::Solver::decideBackendForProblem(problem, options);
    EXPECT_EQ(decision.backend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(decision.reason, "point_only_uses_reference_ba");

    options.refineCameraPose = true;
    options.minPlaMatrixCudaCameras = 50;
    options.minPlaMatrixCudaObservations = 500000;
    options.minPlaMatrixOpenClCameras = 50;
    options.minPlaMatrixOpenClObservations = 500000;
    options.minPlaMatrixDenseCameras = 50;
    options.minPlaMatrixCudaDenseObservations = 500000;
    options.minPlaMatrixOpenClDenseObservations = 500000;
    decision = plabundle::Solver::decideBackendForProblem(problem, options);
    EXPECT_EQ(decision.backend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(decision.reason, "joint_problem_uses_plamatrix_cpu");
}

TEST(PlaBundleBackendSelectionTest, SoftConstraintParticipatesInJointSolverDecision)
{
    plabundle::Problem problem = makeProblem();
    problem.tracks.front().controlPointConstraints.push_back({{0.0, 0.0, 5.0}, 0.1, 1.0, 0});
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    options.refineCameraPose = false;
    options.minPlaMatrixCudaCameras = 1000;
    options.minPlaMatrixCudaObservations = 1000000;
    options.minPlaMatrixOpenClCameras = 1000;
    options.minPlaMatrixOpenClObservations = 1000000;
    options.minPlaMatrixDenseCameras = 1000;

    const plabundle::BackendDecision decision = plabundle::Solver::decideBackendForProblem(problem, options);
    EXPECT_EQ(decision.backend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(decision.reason, "constraint_problem_uses_plamatrix_cpu");
}

TEST(PlaBundleBackendSelectionTest, PolicyThresholdsMatchMeasuredCrossovers)
{
    plabundle::Options options;
    plabundle::ProblemStats stats;
    stats.cameraCount = 96;
    stats.observationCount = 30720;
    EXPECT_FALSE(plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixCuda, stats, options));
    EXPECT_FALSE(
        plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixOpenCl, stats, options));

    stats.cameraCount = 128;
    stats.observationCount = 40960;
    EXPECT_TRUE(plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixCuda, stats, options));
    EXPECT_FALSE(
        plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixOpenCl, stats, options));

    stats.cameraCount = 160;
    stats.observationCount = 51200;
    EXPECT_TRUE(plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixCuda, stats, options));
    EXPECT_TRUE(plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixOpenCl, stats, options));

    stats.cameraCount = 123;
    stats.observationCount = 223593;
    EXPECT_TRUE(plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixCuda, stats, options));
    EXPECT_TRUE(plabundle::Solver::autoBackendMeetsScaleThreshold(plabundle::Backend::PlaMatrixOpenCl, stats, options));
}

TEST(PlaBundleBackendSelectionTest, AutoPrefersAvailableCudaThenOpenCl)
{
    const plabundle::Problem problem = makeProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    options.refineCameraPose = true;
    options.minPlaMatrixCudaCameras = 1;
    options.minPlaMatrixCudaObservations = 1;
    options.minPlaMatrixOpenClCameras = 1;
    options.minPlaMatrixOpenClObservations = 1;

    const plabundle::BackendDecision decision = plabundle::Solver::decideBackendForProblem(problem, options);
    if (plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda, options.plaMatrixDevice))
    {
        EXPECT_EQ(decision.backend, plabundle::Backend::PlaMatrixCuda);
        EXPECT_EQ(decision.reason, "large_joint_problem_uses_plamatrix_cuda");
    }
    else if (plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl, options.plaMatrixDevice))
    {
        EXPECT_EQ(decision.backend, plabundle::Backend::PlaMatrixOpenCl);
        EXPECT_EQ(decision.reason, "large_joint_problem_uses_plamatrix_opencl");
    }
    else
    {
        EXPECT_EQ(decision.backend, plabundle::Backend::PlaMatrixCpu);
        EXPECT_EQ(decision.reason, "joint_problem_uses_plamatrix_cpu");
    }
}

TEST(PlaBundleBackendSelectionTest, CapabilitiesAreIndependentOfRuntimeAvailability)
{
    for (const plabundle::Backend backend : {plabundle::Backend::Auto,
                                             plabundle::Backend::PlaMatrixCpu,
                                             plabundle::Backend::PlaMatrixCuda,
                                             plabundle::Backend::PlaMatrixOpenCl})
    {
        const plabundle::BackendCapabilities capabilities = plabundle::Solver::backendCapabilities(backend);
        EXPECT_TRUE(capabilities.optimizesPoints);
        EXPECT_TRUE(capabilities.refinesCameraPose);
        EXPECT_TRUE(capabilities.refinesSharedFocalLength);
        EXPECT_TRUE(capabilities.refinesSharedFocalAspectRatio);
        EXPECT_TRUE(capabilities.refinesSharedPrincipalPoint);
        EXPECT_TRUE(capabilities.refinesSharedRadialDistortion);
        EXPECT_TRUE(capabilities.supportsSoftConstraints);
        EXPECT_TRUE(capabilities.supportsLaserRangeConstraints);
    }

    EXPECT_FALSE(plabundle::Solver::backendCapabilities(static_cast<plabundle::Backend>(99)).optimizesPoints);
    EXPECT_FALSE(plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda, 9999));
    EXPECT_FALSE(plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl, 9999));
}
