#include <plabundle/solver.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
    plabundle::FrameCamera makeCamera(double centerX,
                                      double centerY,
                                      double focalX = 960.0,
                                      double focalY = 950.0,
                                      double principalX = 512.0,
                                      double principalY = 384.0)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter = {centerX, centerY, 0.0};
        camera.focalXPixels = focalX;
        camera.focalYPixels = focalY;
        camera.principalXPixel = principalX;
        camera.principalYPixel = principalY;
        return camera;
    }

    std::array<double, 2> projectPoint(const plabundle::FrameCamera& camera, const std::array<double, 3>& point)
    {
        plabundle::Projection projection;
        EXPECT_TRUE(plabundle::projectWorldPoint(camera, point, &projection));
        return projection.pixel;
    }

    double distance(const std::array<double, 3>& left, const std::array<double, 3>& right)
    {
        const double dx = left[0] - right[0];
        const double dy = left[1] - right[1];
        const double dz = left[2] - right[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    std::vector<plabundle::Track> makeConstraintTracks(const std::vector<plabundle::FrameCamera>& truthCameras,
                                                       std::vector<std::array<double, 3>>* truthPoints)
    {
        std::vector<plabundle::Track> tracks;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                const std::array<double, 3> truth{
                    {-1.2 + 0.8 * column, -0.8 + 0.8 * row, 10.0 + 0.3 * ((row + column) % 3)}};
                truthPoints->push_back(truth);
                plabundle::Track track;
                track.initialPoint = {truth[0] + 0.12 * ((column % 2) ? 1.0 : -1.0),
                                      truth[1] + 0.09 * ((row % 2) ? -1.0 : 1.0),
                                      truth[2] + 0.18};
                track.controlPointConstraints.push_back({truth, 0.03, 1.0, static_cast<int>(tracks.size())});
                track.laserPlaneConstraints.push_back({truth, {0.0, 0.0, 1.0}, 1.0, 0.0, 0});
                for (std::size_t camera_index = 0; camera_index < truthCameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(truthCameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                tracks.push_back(std::move(track));
            }
        }
        return tracks;
    }

    plabundle::LaserRangeConstraint makeLaserShot(const std::vector<plabundle::FrameCamera>& truthCameras,
                                                  const std::array<double, 3>& truthPoint)
    {
        plabundle::LaserRangeConstraint shot;
        shot.cameraIndex = 2;
        shot.initialPoint = {truthPoint[0] + 0.3, truthPoint[1] - 0.2, truthPoint[2] + 0.25};
        shot.sigmaRangeMeters = 0.03;
        shot.weight = 1.0;
        shot.leverArmCameraMeters = {0.15, -0.05, 0.08};
        const auto center = truthCameras[2].cameraCenter;
        const std::array<double, 3> emitter{{center[0] + shot.leverArmCameraMeters[0],
                                             center[1] + shot.leverArmCameraMeters[1],
                                             center[2] + shot.leverArmCameraMeters[2]}};
        shot.observedRangeMeters = distance(truthPoint, emitter);
        shot.pointMode = plabundle::LaserPointMode::Constrained;
        shot.pointPrior = truthPoint;
        shot.pointPriorSqrtInformation = {5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0};
        shot.shotId = "plabundle-constraint-shot";
        for (int camera_index : {0, 1})
        {
            const auto pixel = projectPoint(truthCameras[static_cast<std::size_t>(camera_index)], truthPoint);
            shot.measuredImageObservations.push_back({camera_index, pixel[0], pixel[1], 1.0, 1.0});
        }
        return shot;
    }

    plabundle::Problem makeFullBrownProblem(const std::vector<plabundle::FrameCamera>& truthCameras,
                                            const std::vector<plabundle::FrameCamera>& initialCameras)
    {
        plabundle::Problem problem;
        problem.cameras = initialCameras;
        for (int row = -4; row <= 4; ++row)
        {
            for (int column = -5; column <= 5; ++column)
            {
                plabundle::Track track;
                track.initialPoint = {0.45 * column, 0.38 * row, 9.0 + 0.35 * ((row + column + 20) % 4)};
                track.controlPointConstraints.push_back(
                    {track.initialPoint, 0.01, 1.0, static_cast<int>(problem.tracks.size())});
                for (std::size_t camera_index = 0; camera_index < truthCameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(truthCameras[camera_index], track.initialPoint);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                problem.tracks.push_back(std::move(track));
            }
        }
        return problem;
    }

    plabundle::Problem makeFilteredTrackProblem()
    {
        plabundle::Problem problem;
        problem.cameras = {makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -3.0), makeCamera(0.0, 3.0)};
        for (int index = 0; index < 6; ++index)
        {
            const std::array<double, 3> truth{
                {-1.0 + 0.4 * index, -0.45 + 0.18 * (index % 3), 8.0 + 0.25 * (index % 2)}};
            plabundle::Track track;
            track.initialPoint = {truth[0] + 0.05, truth[1] - 0.04, truth[2] + 0.12};
            for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
            {
                auto pixel = projectPoint(problem.cameras[camera_index], truth);
                if (index == 5 && camera_index == 3)
                {
                    pixel[0] += 80.0;
                    pixel[1] -= 45.0;
                }
                track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
            }
            problem.tracks.push_back(std::move(track));
        }
        return problem;
    }

    plabundle::Problem makePointOnlyProblem()
    {
        plabundle::Problem problem;
        problem.cameras = {makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -3.0), makeCamera(0.0, 3.0)};
        for (int row = -2; row <= 2; ++row)
        {
            for (int column = -3; column <= 3; ++column)
            {
                const std::array<double, 3> truth{{0.35 * column, 0.32 * row, 8.0 + 0.15 * ((row + column + 10) % 3)}};
                plabundle::Track track;
                track.initialPoint = {truth[0] + 0.12, truth[1] - 0.08, truth[2] + 0.25};
                for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(problem.cameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                problem.tracks.push_back(std::move(track));
            }
        }
        return problem;
    }

    plabundle::Problem makeJointPoseProblem()
    {
        const std::vector<plabundle::FrameCamera> truth_cameras{
            makeCamera(-4.0, 0.0), makeCamera(4.0, 0.0), makeCamera(0.0, -3.5), makeCamera(0.0, 3.5)};
        plabundle::Problem problem;
        problem.cameras = truth_cameras;
        EXPECT_TRUE(plabundle::applyPoseDelta(&problem.cameras[2], {0.008, -0.012, 0.006, 0.22, -0.14, 0.08}));
        EXPECT_TRUE(plabundle::applyPoseDelta(&problem.cameras[3], {-0.006, 0.009, -0.005, -0.18, 0.11, -0.06}));
        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 5; ++column)
            {
                const std::array<double, 3> truth{
                    {-1.6 + 0.8 * column, -1.2 + 0.8 * row, 18.0 + 0.35 * ((row + column) % 3)}};
                plabundle::Track track;
                track.initialPoint = {truth[0] + 0.12 * ((column % 3) - 1),
                                      truth[1] + 0.09 * ((row % 3) - 1),
                                      truth[2] + 0.25 * (((row + column) % 3) - 1)};
                for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(truth_cameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                problem.tracks.push_back(std::move(track));
            }
        }
        problem.tracks.back().observations.back().u += 8.0;
        problem.fixedCameraIndices = {0, 1};
        problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
        return problem;
    }

} // namespace

TEST(PlaBundleSolverTest, SolvesAllSurveyConstraintFamiliesAndPublishesDiagnostics)
{
    std::vector<plabundle::FrameCamera> truth_cameras{
        makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -2.5), makeCamera(0.0, 2.5)};
    for (plabundle::FrameCamera& camera : truth_cameras)
    {
        camera.distortion = {-0.012, 0.0008, -0.00005, 0.0002, -0.00015};
    }
    std::vector<plabundle::FrameCamera> initial_cameras = truth_cameras;
    ASSERT_TRUE(plabundle::applyPoseDelta(&initial_cameras[2], {0.006, -0.008, 0.004, 0.16, -0.1, 0.14}));
    ASSERT_TRUE(plabundle::applyPoseDelta(&initial_cameras[3], {-0.005, 0.007, -0.003, -0.13, 0.09, -0.12}));

    plabundle::Problem problem;
    problem.cameras = initial_cameras;
    std::vector<std::array<double, 3>> truth_points;
    problem.tracks = makeConstraintTracks(truth_cameras, &truth_points);
    problem.fixedCameraIndices = {0, 1};
    problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
    problem.scaleBarConstraints.push_back({0, 1, distance(truth_points[0], truth_points[1]), 0.02, 1.0, 0});
    problem.cameraPosePriors.resize(truth_cameras.size());
    for (std::size_t camera_index : {2U, 3U})
    {
        plabundle::CameraPosePrior prior;
        prior.cameraCenter = truth_cameras[camera_index].cameraCenter;
        prior.cameraToWorldRotation = truth_cameras[camera_index].cameraToWorldRotation;
        prior.positionSigmaMeters = 0.08;
        prior.rotationSigmaDegrees = 1.0;
        problem.cameraPosePriors[camera_index] = prior;
    }
    plabundle::CameraPlaneConstraint plane;
    plane.point = {0.0, 0.0, 0.0};
    plane.normal = {0.0, 0.0, 1.0};
    plane.sigmaMeters = 0.1;
    plane.weight = 10.0;
    problem.cameraPlaneConstraint = plane;
    const std::array<double, 3> laser_truth{{0.35, -0.25, 9.6}};
    problem.laserRangeConstraints.push_back(makeLaserShot(truth_cameras, laser_truth));

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.controlPointWeight = 40.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.laserPlaneWeight = 20.0;
    options.laserHuberDeltaMeters = 0.0;
    options.scaleBarWeight = 50.0;
    options.scaleBarHuberDeltaMeters = 0.0;
    options.cameraPosePriorWeight = 20.0;
    options.cameraPosePriorHuberDelta = 0.0;
    options.cameraPlaneHuberDelta = 0.0;
    options.laserRangeWeight = 10.0;
    options.laserRangeHuberDelta = 0.0;
    options.enablePointFilter = false;
    options.maxIterations = 50;
    options.allowBackendFallback = false;

    const plabundle::Result cpu = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(cpu.usable()) << cpu.backendMessage;
    EXPECT_GT(cpu.quality.controlPointConstraintCount, 0);
    EXPECT_GT(cpu.quality.laserConstraintCount, 0);
    EXPECT_GT(cpu.quality.scaleBarConstraintCount, 0);
    EXPECT_EQ(cpu.quality.laserRangeConstraintCount, 1);
    EXPECT_LT(cpu.quality.meanRmsAfter, cpu.quality.meanRmsBefore);
    EXPECT_LT(cpu.quality.controlPointRmsAfterMeters, cpu.quality.controlPointRmsBeforeMeters);
    EXPECT_LT(cpu.quality.laserRmsAfterMeters, cpu.quality.laserRmsBeforeMeters);
    ASSERT_EQ(cpu.laserRangeShots.size(), 1U);
    EXPECT_TRUE(cpu.laserRangeShots.front().valid);
    EXPECT_EQ(cpu.laserRangeShots.front().shotId, "plabundle-constraint-shot");
    EXPECT_LT(std::abs(cpu.laserRangeShots.front().residualAfterMeters),
              std::abs(cpu.laserRangeShots.front().residualBeforeMeters));

    for (const plabundle::Backend backend : {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl})
    {
        if (!plabundle::Solver::isBackendAvailable(backend, options.plaMatrixDevice))
        {
            continue;
        }
        SCOPED_TRACE(plabundle::backendName(backend));
        options.backend = backend;
        const plabundle::Result accelerated = plabundle::Solver().solve(problem, options);
        ASSERT_TRUE(accelerated.usable()) << accelerated.backendMessage;
        EXPECT_EQ(accelerated.usedBackend, backend);
        EXPECT_TRUE(accelerated.usedGpu);
        EXPECT_TRUE(accelerated.plaMatrix.schurAssemblyOnDevice);
        EXPECT_EQ(accelerated.quality.controlPointConstraintCount, cpu.quality.controlPointConstraintCount);
        EXPECT_EQ(accelerated.quality.laserConstraintCount, cpu.quality.laserConstraintCount);
        EXPECT_EQ(accelerated.quality.scaleBarConstraintCount, cpu.quality.scaleBarConstraintCount);
        EXPECT_EQ(accelerated.quality.laserRangeConstraintCount, cpu.quality.laserRangeConstraintCount);
        EXPECT_TRUE(std::isfinite(accelerated.quality.meanRmsAfter));
        EXPECT_LT(accelerated.quality.meanRmsAfter, accelerated.quality.meanRmsBefore);
        EXPECT_LT(accelerated.plaMatrix.finalCost, accelerated.plaMatrix.initialCost);
        for (const std::size_t camera_index : {2U, 3U})
        {
            EXPECT_LT(distance(accelerated.refinedCameras[camera_index].cameraCenter,
                               cpu.refinedCameras[camera_index].cameraCenter),
                      1.0e-2);
        }
        ASSERT_EQ(accelerated.laserRangeShots.size(), cpu.laserRangeShots.size());
        EXPECT_LT(distance(accelerated.laserRangeShots.front().point, cpu.laserRangeShots.front().point), 1.0e-2);
    }
}

TEST(PlaBundleSolverTest, AcceleratedJointPoseBackendsMatchCpuAndUseDeviceSchur)
{
    const plabundle::Problem problem = makeJointPoseProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 40;
    options.allowBackendFallback = false;
    options.plaMatrixPreconditionerClusterSize = 2;
    const plabundle::Result cpu = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(cpu.usable()) << cpu.backendMessage;

    for (const plabundle::Backend backend : {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl})
    {
        if (!plabundle::Solver::isBackendAvailable(backend, options.plaMatrixDevice))
        {
            continue;
        }
        options.backend = backend;
        const plabundle::Result accelerated = plabundle::Solver().solve(problem, options);
        ASSERT_TRUE(accelerated.usable()) << plabundle::backendName(backend) << ": " << accelerated.backendMessage;
        EXPECT_EQ(accelerated.requestedBackend, backend);
        EXPECT_EQ(accelerated.usedBackend, backend);
        EXPECT_TRUE(accelerated.usedGpu);
        EXPECT_FALSE(accelerated.backendFallback);
        EXPECT_FALSE(accelerated.plaMatrix.deviceName.empty());
        EXPECT_GT(accelerated.plaMatrix.linearIterations, 0);
        EXPECT_GE(accelerated.plaMatrix.schurPatternBuilds, 1);
        EXPECT_GT(accelerated.plaMatrix.schurPatternReuses, 0);
        EXPECT_TRUE(accelerated.plaMatrix.schurAssemblyOnDevice);
        EXPECT_GT(accelerated.plaMatrix.linearSolveSeconds, 0.0);
        EXPECT_DOUBLE_EQ(accelerated.plaMatrix.linearToleranceMinimum, 1.0e-12);
        EXPECT_DOUBLE_EQ(accelerated.plaMatrix.linearToleranceMaximum, 1.0e-12);
        EXPECT_NEAR(accelerated.quality.meanRmsAfter, cpu.quality.meanRmsAfter, 1.0e-5);
        EXPECT_NEAR(accelerated.plaMatrix.finalCost, cpu.plaMatrix.finalCost, 5.0e-4);
        ASSERT_EQ(accelerated.refinedCameras.size(), cpu.refinedCameras.size());
        ASSERT_EQ(accelerated.points.size(), cpu.points.size());
        for (std::size_t index = 0; index < cpu.refinedCameras.size(); ++index)
        {
            EXPECT_LT(distance(accelerated.refinedCameras[index].cameraCenter, cpu.refinedCameras[index].cameraCenter),
                      2.0e-4);
        }
        for (std::size_t index = 0; index < cpu.points.size(); ++index)
        {
            EXPECT_LT(distance(accelerated.points[index].point, cpu.points[index].point), 3.0e-4);
        }
    }
}

TEST(PlaBundleSolverTest, AutoRunsPreferredAvailableAcceleratorForJointProblem)
{
    const plabundle::Problem problem = makeJointPoseProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 40;
    options.minPlaMatrixCudaCameras = 1;
    options.minPlaMatrixCudaObservations = 1;
    options.minPlaMatrixOpenClCameras = 1;
    options.minPlaMatrixOpenClObservations = 1;

    const bool cuda_available = plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda);
    const bool opencl_available = plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl);
    const plabundle::Backend expected_backend = cuda_available     ? plabundle::Backend::PlaMatrixCuda
                                                : opencl_available ? plabundle::Backend::PlaMatrixOpenCl
                                                                   : plabundle::Backend::PlaMatrixCpu;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_EQ(result.requestedBackend, plabundle::Backend::Auto);
    EXPECT_EQ(result.usedBackend, expected_backend);
    EXPECT_FALSE(result.backendFallback);
    EXPECT_FALSE(result.qualityGateRejected);
    EXPECT_NE(result.backendSelectionReason.find("quality_gate_passed"), std::string::npos);
    if (expected_backend == plabundle::Backend::PlaMatrixCpu)
    {
        EXPECT_FALSE(result.usedGpu);
    }
    else
    {
        EXPECT_TRUE(result.usedGpu);
        EXPECT_TRUE(result.plaMatrix.schurAssemblyOnDevice);
        EXPECT_FALSE(result.plaMatrix.deviceName.empty());
    }
}

TEST(PlaBundleSolverTest, OpenClSelectedDeviceHonorsIndexAndFp64Contract)
{
    int requested_index = -1;
    if (const char* environment_index = std::getenv("PLAMATRIX_OPENCL_DEVICE_INDEX");
        environment_index && environment_index[0] != '\0' && std::string(environment_index) != "-1")
    {
        char* parse_end = nullptr;
        const long parsed_index = std::strtol(environment_index, &parse_end, 10);
        if (!parse_end || parse_end[0] != '\0' || parsed_index < 0 || parsed_index > std::numeric_limits<int>::max())
        {
            GTEST_SKIP() << "PLAMATRIX_OPENCL_DEVICE_INDEX is not a valid non-negative integer";
        }
        requested_index = static_cast<int>(parsed_index);
    }
    else
    {
        for (int candidate = 0; candidate < 16; ++candidate)
        {
            if (plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl, candidate))
            {
                requested_index = candidate;
                break;
            }
        }
    }

    if (requested_index < 0)
    {
        GTEST_SKIP() << "PlaMatrix OpenCL backend is unavailable";
    }

    const bool available = plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl, requested_index);
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixOpenCl;
    options.plaMatrixDevice = requested_index;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 20;
    options.allowBackendFallback = false;
    const plabundle::Result result = plabundle::Solver().solve(makeJointPoseProblem(), options);

    EXPECT_EQ(result.requestedBackend, plabundle::Backend::PlaMatrixOpenCl);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixOpenCl);
    if (available)
    {
        ASSERT_TRUE(result.usable()) << result.backendMessage;
        EXPECT_TRUE(result.usedGpu);
        EXPECT_TRUE(result.plaMatrix.schurAssemblyOnDevice);
        EXPECT_FALSE(result.plaMatrix.deviceName.empty());
    }
    else
    {
        EXPECT_EQ(result.status, plabundle::SolveStatus::BackendUnavailable);
        EXPECT_FALSE(result.usable());
        EXPECT_FALSE(result.usedGpu);
        EXPECT_FALSE(result.backendMessage.empty());
    }
}

TEST(PlaBundleSolverTest, RecoversSharedFullBrownCalibration)
{
    std::vector<plabundle::FrameCamera> truth_cameras{makeCamera(-3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, -2.5, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, 2.5, 1040.0, 998.4, 516.0, 381.5)};
    for (plabundle::FrameCamera& camera : truth_cameras)
    {
        camera.distortion = {-0.035, 0.004, -0.0004, 0.0008, -0.0006};
    }
    const std::vector<plabundle::FrameCamera> initial_cameras{makeCamera(-3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(0.0, -2.5, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(0.0, 2.5, 900.0, 900.0, 512.0, 384.0)};
    const plabundle::Problem problem = makeFullBrownProblem(truth_cameras, initial_cameras);

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    options.controlPointWeight = 10000.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.sharedFocalPriorSigma = 5.0;
    options.sharedFocalAspectPriorSigma = 5.0;
    options.sharedPrincipalPointPriorSigmaFraction = 1.0;
    options.sharedRadialK1PriorSigma = 5.0;
    options.sharedRadialK2PriorSigma = 5.0;
    options.sharedRadialK3PriorSigma = 5.0;
    options.sharedTangentialP1PriorSigma = 5.0;
    options.sharedTangentialP2PriorSigma = 5.0;
    options.enablePointFilter = false;
    options.maxIterations = 60;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), truth_cameras.size());
    EXPECT_EQ(result.quality.refinedCalibrationGroupCount, 1);
    EXPECT_EQ(result.quality.selfCalibrationStagesRun, 1);
    EXPECT_EQ(result.quality.refinedIntrinsicCount, static_cast<int>(truth_cameras.size()));
    EXPECT_LT(result.quality.meanRmsAfter, result.quality.meanRmsBefore);
    EXPECT_NEAR(result.refinedCameras.front().focalXPixels, 1040.0, 30.0);
    EXPECT_NEAR(result.refinedCameras.front().focalYPixels, 998.4, 30.0);
    EXPECT_NEAR(result.refinedCameras.front().principalXPixel, 516.0, 8.0);
    EXPECT_NEAR(result.refinedCameras.front().principalYPixel, 381.5, 8.0);
    EXPECT_NEAR(result.refinedCameras.front().distortion.k1, -0.035, 0.02);
    EXPECT_NEAR(result.refinedCameras.front().distortion.p1, 0.0008, 0.001);
    EXPECT_NEAR(result.refinedCameras.front().distortion.p2, -0.0006, 0.001);
}

TEST(PlaBundleSolverTest, CudaMixedPrecisionOptionPreservesFullBrownParity)
{
    if (!plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda))
    {
        GTEST_SKIP() << "PlaMatrix CUDA backend is unavailable";
    }

    std::vector<plabundle::FrameCamera> truth_cameras{makeCamera(-3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, -2.5, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, 2.5, 1040.0, 998.4, 516.0, 381.5)};
    for (plabundle::FrameCamera& camera : truth_cameras)
    {
        camera.distortion = {-0.035, 0.004, -0.0004, 0.0008, -0.0006};
    }
    const std::vector<plabundle::FrameCamera> initial_cameras{makeCamera(-3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(0.0, -2.5, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(0.0, 2.5, 900.0, 900.0, 512.0, 384.0)};
    const plabundle::Problem problem = makeFullBrownProblem(truth_cameras, initial_cameras);

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    options.controlPointWeight = 10000.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.sharedFocalPriorSigma = 5.0;
    options.sharedFocalAspectPriorSigma = 5.0;
    options.sharedPrincipalPointPriorSigmaFraction = 1.0;
    options.sharedRadialK1PriorSigma = 5.0;
    options.sharedRadialK2PriorSigma = 5.0;
    options.sharedRadialK3PriorSigma = 5.0;
    options.sharedTangentialP1PriorSigma = 5.0;
    options.sharedTangentialP2PriorSigma = 5.0;
    options.enablePointFilter = false;
    options.maxIterations = 60;
    options.allowBackendFallback = false;
    const plabundle::Result cpu = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(cpu.usable()) << cpu.backendMessage;

    options.backend = plabundle::Backend::PlaMatrixCuda;
    options.enablePlaMatrixMixedPrecision = true;
    const plabundle::Result cuda = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(cuda.usable()) << cuda.backendMessage;
    EXPECT_TRUE(cuda.usedGpu);
    EXPECT_TRUE(cuda.plaMatrix.schurAssemblyOnDevice);
    // mixedPrecisionUsed reports an accepted FP32 seed, not merely that the option was requested. Strict problems may
    // legitimately reject that seed and continue in FP64, so the stable contract here is device execution and parity.
    EXPECT_NEAR(cuda.quality.meanRmsAfter, cpu.quality.meanRmsAfter, 2.0e-3);
    EXPECT_NEAR(cuda.refinedCameras.front().focalXPixels, cpu.refinedCameras.front().focalXPixels, 2.0);
    EXPECT_NEAR(cuda.refinedCameras.front().focalYPixels, cpu.refinedCameras.front().focalYPixels, 2.0);
    EXPECT_NEAR(cuda.refinedCameras.front().principalXPixel, cpu.refinedCameras.front().principalXPixel, 0.5);
    EXPECT_NEAR(cuda.refinedCameras.front().principalYPixel, cpu.refinedCameras.front().principalYPixel, 0.5);
    EXPECT_NEAR(cuda.refinedCameras.front().distortion.k1, cpu.refinedCameras.front().distortion.k1, 2.0e-3);
    EXPECT_NEAR(cuda.refinedCameras.front().distortion.k2, cpu.refinedCameras.front().distortion.k2, 3.0e-3);
    EXPECT_NEAR(cuda.refinedCameras.front().distortion.k3, cpu.refinedCameras.front().distortion.k3, 3.0e-3);
}

TEST(PlaBundleSolverTest, RefinesIndependentSharedFocalGroups)
{
    const std::vector<plabundle::FrameCamera> truth_cameras{makeCamera(-3.0, 0.0, 900.0, 900.0),
                                                            makeCamera(3.0, 0.0, 900.0, 900.0),
                                                            makeCamera(0.0, -2.5, 1200.0, 1200.0),
                                                            makeCamera(0.0, 2.5, 1200.0, 1200.0)};
    const std::vector<plabundle::FrameCamera> initial_cameras{makeCamera(-3.0, 0.0, 1000.0, 1000.0),
                                                              makeCamera(3.0, 0.0, 1000.0, 1000.0),
                                                              makeCamera(0.0, -2.5, 1000.0, 1000.0),
                                                              makeCamera(0.0, 2.5, 1000.0, 1000.0)};
    plabundle::Problem problem = makeFullBrownProblem(truth_cameras, initial_cameras);
    problem.cameraCalibrationGroupIds = {0, 0, 1, 1};

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.controlPointWeight = 10000.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.sharedFocalPriorSigma = 5.0;
    options.enablePointFilter = false;
    options.maxIterations = 40;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), truth_cameras.size());
    EXPECT_EQ(result.quality.refinedCalibrationGroupCount, 2);
    EXPECT_NEAR(result.refinedCameras[0].focalXPixels, result.refinedCameras[1].focalXPixels, 1.0e-10);
    EXPECT_NEAR(result.refinedCameras[2].focalXPixels, result.refinedCameras[3].focalXPixels, 1.0e-10);
    EXPECT_NEAR(result.refinedCameras[0].focalXPixels, 900.0, 20.0);
    EXPECT_NEAR(result.refinedCameras[2].focalXPixels, 1200.0, 20.0);
    EXPECT_GT(result.refinedCameras[2].focalXPixels - result.refinedCameras[0].focalXPixels, 250.0);
}

TEST(PlaBundleSolverTest, RejectsExtendedIntrinsicsWithoutSharedFocal)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-1.0, 0.0), makeCamera(1.0, 0.0)};
    plabundle::Track track;
    track.initialPoint = {0.0, 0.0, 8.0};
    for (std::size_t index = 0; index < problem.cameras.size(); ++index)
    {
        const auto pixel = projectPoint(problem.cameras[index], track.initialPoint);
        track.observations.push_back({static_cast<int>(index), pixel[0], pixel[1], 1.0, 1.0});
    }
    problem.tracks.push_back(track);

    plabundle::Options options;
    options.refineCameraPose = false;
    options.refineSharedPrincipalPoint = true;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::InvalidInput);
    EXPECT_FALSE(result.usable());
}

TEST(PlaBundleSolverTest, ReferenceOnlineSchurMatchesGeneralSchurPath)
{
    std::vector<plabundle::FrameCamera> truth_cameras{makeCamera(-3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, -2.5, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, 2.5, 1040.0, 998.4, 516.0, 381.5)};
    for (plabundle::FrameCamera& camera : truth_cameras)
    {
        camera.distortion = {-0.035, 0.004, -0.0004, 0.0008, -0.0006};
    }
    const std::vector<plabundle::FrameCamera> initial_cameras{makeCamera(-3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(0.0, -2.5, 900.0, 900.0, 512.0, 384.0),
                                                              makeCamera(0.0, 2.5, 900.0, 900.0, 512.0, 384.0)};
    plabundle::Problem problem = makeFullBrownProblem(truth_cameras, initial_cameras);
    for (plabundle::Track& track : problem.tracks)
    {
        track.controlPointConstraints.clear();
    }

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    options.sharedFocalPriorSigma = 5.0;
    options.sharedFocalAspectPriorSigma = 5.0;
    options.sharedPrincipalPointPriorSigmaFraction = 1.0;
    options.sharedRadialK1PriorSigma = 5.0;
    options.sharedRadialK2PriorSigma = 5.0;
    options.sharedRadialK3PriorSigma = 5.0;
    options.sharedTangentialP1PriorSigma = 5.0;
    options.sharedTangentialP2PriorSigma = 5.0;
    options.enablePointFilter = false;
    options.maxIterations = 20;

    const plabundle::Result general = plabundle::Solver().solve(problem, options);
    options.useReferenceOnlineSchur = true;
    const plabundle::Result reference = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(general.usable()) << general.backendMessage;
    ASSERT_TRUE(reference.usable()) << reference.backendMessage;
    EXPECT_TRUE(reference.plaMatrix.referenceOnlineSchurUsed);
    EXPECT_FALSE(general.plaMatrix.referenceOnlineSchurUsed);
    EXPECT_NEAR(reference.plaMatrix.finalCost, general.plaMatrix.finalCost, 1.0e-4);
    EXPECT_NEAR(reference.quality.meanRmsAfter, general.quality.meanRmsAfter, 1.0e-5);
    EXPECT_NEAR(reference.refinedCameras.front().focalXPixels, general.refinedCameras.front().focalXPixels, 1.0e-3);
    EXPECT_NEAR(reference.refinedCameras.front().distortion.k1, general.refinedCameras.front().distortion.k1, 1.0e-5);
}

TEST(PlaBundleSolverTest, FixedTrackRemainsUnchangedWhileFreeTrackIsRefined)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-1.0, 0.0), makeCamera(1.0, 0.0)};
    const std::array<double, 3> truth{{0.0, 0.0, 5.0}};
    const std::array<double, 3> fixed_initial{{0.3, -0.2, 6.0}};
    const std::array<double, 3> free_initial{{-0.4, 0.3, 6.5}};
    for (const std::array<double, 3>& initial : {fixed_initial, free_initial})
    {
        plabundle::Track track;
        track.initialPoint = initial;
        for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
        {
            const auto pixel = projectPoint(problem.cameras[camera_index], truth);
            track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
        }
        problem.tracks.push_back(std::move(track));
    }
    problem.fixedTrackIndices = {0};

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 3;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.points.size(), problem.tracks.size());
    EXPECT_EQ(result.points[0].point, fixed_initial);
    EXPECT_NE(result.points[1].point, free_initial);
    EXPECT_LT(distance(result.points[1].point, truth), distance(free_initial, truth));
}

TEST(PlaBundleSolverTest, ExplicitGaugeRequiresAndHonorsFixedCameraBlocks)
{
    plabundle::Problem problem = makePointOnlyProblem();
    problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 8;

    const plabundle::Result rejected = plabundle::Solver().solve(problem, options);
    EXPECT_EQ(rejected.status, plabundle::SolveStatus::UnsupportedConfiguration);
    EXPECT_FALSE(rejected.usable());

    problem.fixedCameraIndices = {0, 1};
    const plabundle::Result solved = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(solved.usable()) << solved.backendMessage;
    ASSERT_EQ(solved.refinedCameras.size(), problem.cameras.size());
    EXPECT_EQ(solved.refinedCameras[0].cameraCenter, problem.cameras[0].cameraCenter);
    EXPECT_EQ(solved.refinedCameras[0].cameraToWorldRotation, problem.cameras[0].cameraToWorldRotation);
    EXPECT_EQ(solved.refinedCameras[1].cameraCenter, problem.cameras[1].cameraCenter);
    EXPECT_EQ(solved.refinedCameras[1].cameraToWorldRotation, problem.cameras[1].cameraToWorldRotation);
}

TEST(PlaBundleSolverTest, FiltersGrossTrackWithAProblemWideThreshold)
{
    const plabundle::Problem problem = makeFilteredTrackProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = true;
    options.filterMaxReprojError = 0.5;
    options.filterSigmaFactor = 2.0;
    options.maxIterations = 10;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.points.size(), problem.tracks.size());
    for (std::size_t index = 0; index + 1 < result.points.size(); ++index)
    {
        EXPECT_TRUE(result.points[index].valid) << index;
    }
    EXPECT_FALSE(result.points.back().valid);
    EXPECT_EQ(result.quality.optimizedTracks, 5);
    EXPECT_DOUBLE_EQ(result.quality.validTrackRatio, 5.0 / 6.0);
}

TEST(PlaBundleSolverTest, AutoQualityGateRejectsLowValidTrackRatio)
{
    const plabundle::Problem problem = makeFilteredTrackProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    options.refineCameraPose = false;
    options.enablePointFilter = true;
    options.filterMaxReprojError = 0.5;
    options.filterSigmaFactor = 2.0;
    options.minAcceptedValidTrackRatio = 1.0;
    options.maxIterations = 10;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    EXPECT_EQ(result.requestedBackend, plabundle::Backend::Auto);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(result.status, plabundle::SolveStatus::NumericalFailure);
    EXPECT_FALSE(result.usable());
    EXPECT_TRUE(result.qualityGateRejected);
    EXPECT_NE(result.qualityGateMessage.find("valid-track ratio"), std::string::npos);
    EXPECT_EQ(result.quality.optimizedTracks, 5);
}

TEST(PlaBundleSolverTest, AutoQualityGateFallsBackToCpuAfterAcceleratedRejection)
{
    if (!plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda) &&
        !plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl))
    {
        GTEST_SKIP() << "no accelerated PlaMatrix backend is available";
    }

    plabundle::Problem problem = makeFilteredTrackProblem();
    problem.tracks.back().observations.back().u += 720.0;
    problem.tracks.back().observations.back().v -= 405.0;
    problem.fixedCameraIndices = {0, 1, 3};
    problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
    plabundle::Options options;
    options.backend = plabundle::Backend::Auto;
    options.refineCameraPose = true;
    options.enablePointFilter = true;
    options.filterMaxReprojError = 0.5;
    options.filterSigmaFactor = 2.0;
    options.minAcceptedValidTrackRatio = 1.0;
    options.maxIterations = 10;
    options.minPlaMatrixCudaCameras = 1;
    options.minPlaMatrixCudaObservations = 1;
    options.minPlaMatrixOpenClCameras = 1;
    options.minPlaMatrixOpenClObservations = 1;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_EQ(result.requestedBackend, plabundle::Backend::Auto);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_FALSE(result.usedGpu);
    EXPECT_TRUE(result.qualityGateRejected);
    EXPECT_NE(result.qualityGateMessage.find("valid-track ratio"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("accelerated_candidate_quality_gate_rejected"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("fallback_to_plamatrix_cpu"), std::string::npos);
    EXPECT_EQ(result.quality.optimizedTracks, 5);
}

TEST(PlaBundleSolverTest, CancellationDoesNotPublishIntermediateState)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 12.0}};
    const std::array<double, 3> initial{{0.8, -0.5, 13.5}};
    plabundle::Track track;
    track.initialPoint = initial;
    for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
    {
        const auto pixel = projectPoint(problem.cameras[camera_index], truth);
        track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
    }
    problem.tracks.push_back(track);

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.progressCallback = [](const plabundle::IterationSummary&) { return false; };

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    EXPECT_EQ(result.status, plabundle::SolveStatus::Cancelled);
    EXPECT_FALSE(result.usable());
    ASSERT_EQ(result.points.size(), 1U);
    EXPECT_EQ(result.points.front().point, initial);
    ASSERT_EQ(result.refinedCameras.size(), problem.cameras.size());
    EXPECT_EQ(result.refinedCameras[0].cameraCenter, problem.cameras[0].cameraCenter);
    EXPECT_EQ(result.refinedCameras[1].cameraCenter, problem.cameras[1].cameraCenter);
}

TEST(PlaBundleSolverTest, CpuResultIsDeterministicAcrossThreadCounts)
{
    const plabundle::Problem problem = makePointOnlyProblem();
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 8;
    options.numThreads = 1;
    const plabundle::Result serial = plabundle::Solver().solve(problem, options);
    options.numThreads = 4;
    const plabundle::Result parallel = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(serial.usable()) << serial.backendMessage;
    ASSERT_TRUE(parallel.usable()) << parallel.backendMessage;
    ASSERT_EQ(serial.points.size(), parallel.points.size());
    EXPECT_EQ(serial.status, parallel.status);
    EXPECT_NEAR(serial.plaMatrix.finalCost, parallel.plaMatrix.finalCost, 1.0e-10);
    EXPECT_NEAR(serial.quality.meanRmsAfter, parallel.quality.meanRmsAfter, 1.0e-10);
    for (std::size_t index = 0; index < serial.points.size(); ++index)
    {
        EXPECT_EQ(serial.points[index].valid, parallel.points[index].valid);
        EXPECT_NEAR(distance(serial.points[index].point, parallel.points[index].point), 0.0, 1.0e-10);
    }
}
