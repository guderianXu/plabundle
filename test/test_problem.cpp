#include <plabundle/options.h>
#include <plabundle/problem.h>
#include <plabundle/solver.h>

#include <gtest/gtest.h>

#include <array>
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

TEST(PlaBundleProblemTest, ValidatesRigTopologyAndComposesBoundCameras)
{
    plabundle::Problem problem = makeProblem();
    const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    problem.rig.captures = {{0, 0, identity, {-2.0, 0.0, 0.0}, true}, {0, 1, identity, {2.0, 0.0, 0.0}, false}};
    problem.rig.sensors = {{0, 0, identity, {-0.25, 0.0, 0.0}, true}};
    problem.rig.cameraBindings = {{0, 0, 0, 0}, {1, 0, 1, 0}};
    std::vector<plabundle::FrameCamera> composed;
    std::string error;
    ASSERT_TRUE(plabundle::composeRigCameras(problem.cameras, problem.rig, &composed, &error)) << error;
    ASSERT_EQ(composed.size(), 2U);
    EXPECT_DOUBLE_EQ(composed[0].cameraCenter[0], -2.25);
    EXPECT_DOUBLE_EQ(composed[1].cameraCenter[0], 1.75);
    EXPECT_TRUE(plabundle::validateProblem(problem, &error)) << error;

    problem.rig.cameraBindings.pop_back();
    EXPECT_FALSE(plabundle::validateProblem(problem, &error));
    EXPECT_NE(error.find("exactly one binding"), std::string::npos);
}

TEST(PlaBundleProblemTest, RejectsRigWithoutFixedBodyFrameSensor)
{
    plabundle::Problem problem = makeProblem();
    const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    problem.rig.captures = {{0, 0, identity, {-2.0, 0.0, 0.0}, true}, {0, 1, identity, {2.0, 0.0, 0.0}, false}};
    problem.rig.sensors = {{0, 0, identity, {0.0, 0.0, 0.0}, false}};
    problem.rig.cameraBindings = {{0, 0, 0, 0}, {1, 0, 1, 0}};
    std::string error;
    EXPECT_FALSE(plabundle::validateProblem(problem, &error));
    EXPECT_NE(error.find("fixed sensor extrinsic"), std::string::npos);
}

TEST(PlaBundleProblemTest, RejectsInvalidPosePriorCovarianceFrameAndRank)
{
    plabundle::CameraPosePrior prior;
    prior.uncertainty = plabundle::PosePriorUncertainty::Covariance;
    prior.uncertaintyMatrix.fill(0.0);
    std::string error;
    EXPECT_FALSE(plabundle::validateCameraPosePrior(prior, &error));
    EXPECT_NE(error.find("positive definite"), std::string::npos);

    for (int index = 0; index < 6; ++index)
    {
        prior.uncertaintyMatrix[static_cast<std::size_t>(index * 6 + index)] = 1.0;
    }
    prior.uncertaintyMatrix[1] = 0.25;
    EXPECT_FALSE(plabundle::validateCameraPosePrior(prior, &error));
    EXPECT_NE(error.find("symmetric"), std::string::npos);

    prior.uncertaintyMatrix[1] = 0.0;
    prior.tangentFrame = static_cast<plabundle::PosePriorTangentFrame>(99);
    EXPECT_FALSE(plabundle::validateCameraPosePrior(prior, &error));
    EXPECT_NE(error.find("tangent frame"), std::string::npos);
}

TEST(PlaBundleProblemTest, ValidatesControlPointCovarianceAndRejectsInvalidMatrices)
{
    plabundle::ControlPointConstraint constraint;
    constraint.point = {0.2, -0.1, 5.0};
    constraint.uncertainty = plabundle::ControlPointUncertainty::Covariance;
    constraint.uncertaintyMatrix = {0.04, 0.01, 0.0, 0.01, 0.09, 0.0, 0.0, 0.0, 0.16};
    std::string error;
    EXPECT_TRUE(plabundle::validateControlPointConstraint(constraint, &error)) << error;

    constraint.uncertaintyMatrix[1] = 0.02;
    EXPECT_FALSE(plabundle::validateControlPointConstraint(constraint, &error));
    EXPECT_NE(error.find("symmetric"), std::string::npos);

    constraint.uncertaintyMatrix = {1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0};
    EXPECT_FALSE(plabundle::validateControlPointConstraint(constraint, &error));
    EXPECT_NE(error.find("positive definite"), std::string::npos);

    constraint.uncertainty = plabundle::ControlPointUncertainty::SqrtInformation;
    constraint.uncertaintyMatrix = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    EXPECT_FALSE(plabundle::validateControlPointConstraint(constraint, &error));
    EXPECT_NE(error.find("positive definite"), std::string::npos);

    constraint.uncertaintyMatrix = {2.0, 0.1, 0.0, 0.0, 1.5, -0.2, 0.0, 0.0, 1.0};
    EXPECT_TRUE(plabundle::validateControlPointConstraint(constraint, &error)) << error;

    constraint.uncertaintyMatrix[4] = 0.0;

    plabundle::Problem problem = makeProblem();
    problem.tracks[0].controlPointConstraints.push_back(constraint);
    EXPECT_FALSE(plabundle::validateProblem(problem, &error));
    EXPECT_NE(error.find("control-point"), std::string::npos);
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

TEST(PlaBundleProblemTest, StructuredAndCompatibilityOptionsProduceEquivalentResults)
{
    plabundle::SolveOptions structured;
    structured.backend.requested = plabundle::Backend::PlaMatrixCpu;
    structured.calibration.refineCameraPose = false;
    structured.solver.enablePointFilter = false;
    structured.solver.logIterationProgress = false;

    const plabundle::Options compatibility = plabundle::makeCompatibilityOptions(structured);
    const plabundle::Solver solver;
    const plabundle::Result structured_result = solver.solve(makeProblem(), structured);
    const plabundle::Result compatibility_result = solver.solve(makeProblem(), compatibility);

    ASSERT_TRUE(structured_result.usable());
    ASSERT_TRUE(compatibility_result.usable());
    EXPECT_EQ(structured_result.status, compatibility_result.status);
    EXPECT_EQ(structured_result.requestedBackend, compatibility_result.requestedBackend);
    EXPECT_EQ(structured_result.usedBackend, compatibility_result.usedBackend);
    ASSERT_EQ(structured_result.points.size(), compatibility_result.points.size());
    EXPECT_EQ(structured_result.points[0].point, compatibility_result.points[0].point);
    EXPECT_DOUBLE_EQ(structured_result.quality.meanRmsAfter, compatibility_result.quality.meanRmsAfter);
}

TEST(PlaBundleProblemTest, AcceleratedBackendsRejectInvalidDeviceWithoutFallback)
{
    for (const plabundle::Backend backend :
         {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixVulkan, plabundle::Backend::PlaMatrixOpenCl})
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
    for (const plabundle::Backend backend :
         {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixVulkan, plabundle::Backend::PlaMatrixOpenCl})
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

    options.refineSharedMetashapeParameters = true;
    options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::TangentialP4)] = true;
    EXPECT_TRUE(plabundle::sharedIntrinsicParameterEnabled(options, plabundle::IntrinsicParameter::TangentialP4));
    EXPECT_FALSE(plabundle::sharedIntrinsicParameterEnabled(options, plabundle::IntrinsicParameter::SkewB2));
}
