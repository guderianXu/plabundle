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

    std::array<double, 2> projectRollingPoint(const plabundle::FrameCamera& camera, const std::array<double, 3>& point)
    {
        double line = camera.rollingShutter.referenceLinePixels;
        plabundle::Projection projection;
        for (int iteration = 0; iteration < 12; ++iteration)
        {
            EXPECT_TRUE(plabundle::projectWorldPointAtLine(camera, point, line, &projection));
            line = projection.pixel[1];
        }
        EXPECT_TRUE(plabundle::projectWorldPointAtLine(camera, point, line, &projection));
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

    struct MetashapeExtensionFixture
    {
        plabundle::Problem problem;
        plabundle::MetashapeFrameCalibration truthCalibration;
    };

    MetashapeExtensionFixture makeMetashapeExtensionFixture()
    {
        MetashapeExtensionFixture fixture;
        fixture.truthCalibration = {
            1000.0, 512.0, 384.0, 30.0, 12.0, -0.03, 0.006, -0.001, 0.015, 0.003, -0.002, 0.12, -0.08};
        plabundle::MetashapeFrameCalibration initial_calibration = fixture.truthCalibration;
        initial_calibration.b2 = 0.0;
        initial_calibration.k4 = 0.0;
        initial_calibration.p3 = 0.0;
        initial_calibration.p4 = 0.0;
        std::vector<plabundle::FrameCamera> truth_cameras{
            makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -2.5), makeCamera(0.0, 2.5)};
        std::vector<plabundle::FrameCamera> initial_cameras = truth_cameras;
        for (plabundle::FrameCamera& camera : truth_cameras)
        {
            EXPECT_TRUE(plabundle::applyMetashapeFrameCalibration(&camera, fixture.truthCalibration));
        }
        for (plabundle::FrameCamera& camera : initial_cameras)
        {
            EXPECT_TRUE(plabundle::applyMetashapeFrameCalibration(&camera, initial_calibration));
        }
        fixture.problem = makeFullBrownProblem(truth_cameras, initial_cameras);
        return fixture;
    }

    plabundle::Options makeMetashapeExtensionOptions(plabundle::Backend backend)
    {
        plabundle::Options options;
        options.backend = backend;
        options.refineCameraPose = false;
        options.refineSharedFocalLength = true;
        options.refineSharedMetashapeParameters = true;
        options.useSharedIntrinsicParameterMask = true;
        options.sharedIntrinsicParameterMask.fill(false);
        options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::FocalLength)] =
            true;
        options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::SkewB2)] = true;
        options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::RadialK4)] = true;
        options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::TangentialP3)] =
            true;
        options.sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::TangentialP4)] =
            true;
        options.controlPointWeight = 10000.0;
        options.controlPointHuberDeltaMeters = 0.0;
        options.sharedFocalPriorSigma = 5.0;
        options.sharedSkewPriorSigmaFraction = 1.0;
        options.sharedRadialK4PriorSigma = 5.0;
        options.sharedTangentialP3PriorSigma = 5.0;
        options.sharedTangentialP4PriorSigma = 5.0;
        options.enablePointFilter = false;
        options.maxIterations = 80;
        options.allowBackendFallback = false;
        return options;
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

    plabundle::Problem makeQualityDegradingConstraintProblem()
    {
        plabundle::Problem problem;
        problem.cameras = {makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -2.5), makeCamera(0.0, 2.5)};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                const std::array<double, 3> truth{
                    {-1.2 + 0.8 * column, -0.8 + 0.8 * row, 10.0 + 0.2 * ((row + column) % 3)}};
                plabundle::Track track;
                track.initialPoint = {truth[0] + 1.0e-3, truth[1] - 8.0e-4, truth[2] + 1.5e-3};
                for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(problem.cameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                const double sign = (row + column) % 2 == 0 ? 1.0 : -1.0;
                track.controlPointConstraints.push_back(
                    {{truth[0] + 0.35 * sign, truth[1] - 0.2 * sign, truth[2] + 0.3},
                     1.0e-3,
                     1.0,
                     static_cast<int>(problem.tracks.size())});
                problem.tracks.push_back(std::move(track));
            }
        }
        problem.fixedCameraIndices = {0, 1};
        problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
        return problem;
    }

    plabundle::Options makeQualityDegradingConstraintOptions(plabundle::Backend backend)
    {
        plabundle::Options options;
        options.backend = backend;
        options.refineCameraPose = true;
        options.enablePointFilter = false;
        options.controlPointWeight = 100.0;
        options.controlPointHuberDeltaMeters = 0.0;
        options.maxAcceptedRmsGrowth = 1.0;
        options.maxIterations = 20;
        options.allowBackendFallback = false;
        return options;
    }

    plabundle::Problem makeRigProblem(plabundle::RigTopology* truthRig)
    {
        plabundle::Problem problem;
        problem.cameras.assign(6, makeCamera(0.0, 0.0));
        const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        truthRig->captures = {{0, 0, identity, {-4.0, 0.0, 0.0}, true},
                              {0, 1, identity, {0.0, 0.0, 0.0}, false},
                              {0, 2, identity, {4.0, 0.0, 0.0}, false}};
        truthRig->sensors = {{0, 0, identity, {-0.45, 0.0, 0.0}, true}, {0, 1, identity, {0.45, 0.15, 0.02}, false}};
        for (int capture = 0; capture < 3; ++capture)
        {
            for (int sensor = 0; sensor < 2; ++sensor)
            {
                truthRig->cameraBindings.push_back({capture * 2 + sensor, 0, capture, sensor});
            }
        }
        std::vector<plabundle::FrameCamera> truth_cameras;
        EXPECT_TRUE(plabundle::composeRigCameras(problem.cameras, *truthRig, &truth_cameras));

        problem.rig = *truthRig;
        problem.rig.captures[1].rigCenterInWorld = {0.28, -0.16, 0.1};
        problem.rig.sensors[1].cameraCenterInRig = {0.63, 0.03, 0.1};
        for (int row = 0; row < 5; ++row)
        {
            for (int column = 0; column < 6; ++column)
            {
                const std::array<double, 3> truth{
                    {-1.5 + 0.6 * column, -1.0 + 0.5 * row, 14.0 + 0.25 * ((row + column) % 3)}};
                plabundle::Track track;
                track.initialPoint = {truth[0] + 0.08, truth[1] - 0.05, truth[2] + 0.12};
                for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
                {
                    if (camera_index == 3)
                    {
                        continue;
                    }
                    const auto pixel = projectPoint(truth_cameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                problem.tracks.push_back(std::move(track));
            }
        }
        return problem;
    }

    plabundle::Problem makeMultiRigProblem(plabundle::RigTopology* truthRig)
    {
        plabundle::Problem problem;
        problem.cameras.assign(4, makeCamera(0.0, 0.0));
        const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        truthRig->captures = {{0, 0, identity, {-4.0, 0.0, 0.0}, true},
                              {0, 1, identity, {-1.5, 0.3, 0.0}, false},
                              {1, 0, identity, {1.5, -0.3, 0.0}, false},
                              {1, 1, identity, {4.0, 0.0, 0.0}, false}};
        truthRig->sensors = {{0, 0, identity, {0.0, 0.0, 0.0}, true}, {1, 0, identity, {0.0, 0.0, 0.0}, true}};
        truthRig->cameraBindings = {{0, 0, 0, 0}, {1, 0, 1, 0}, {2, 1, 0, 0}, {3, 1, 1, 0}};
        std::vector<plabundle::FrameCamera> truth_cameras;
        EXPECT_TRUE(plabundle::composeRigCameras(problem.cameras, *truthRig, &truth_cameras));
        problem.rig = *truthRig;
        problem.rig.captures[1].rigCenterInWorld = {-1.18, 0.12, 0.08};
        problem.rig.captures[2].rigCenterInWorld = {1.24, -0.08, -0.06};
        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 5; ++column)
            {
                const std::array<double, 3> truth{
                    {-1.2 + 0.6 * column, -0.8 + 0.55 * row, 13.0 + 0.2 * ((row + column) % 3)}};
                plabundle::Track track;
                track.initialPoint = {truth[0] + 0.06, truth[1] - 0.04, truth[2] + 0.1};
                for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(truth_cameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                problem.tracks.push_back(std::move(track));
            }
        }
        return problem;
    }

    struct RigSurveyFixture
    {
        plabundle::Problem problem;
        plabundle::RigTopology truthRig;
        std::vector<plabundle::FrameCamera> truthCameras;
        std::vector<std::array<double, 3>> truthPoints;
    };

    RigSurveyFixture makeRigSurveyFixture()
    {
        RigSurveyFixture fixture;
        const std::array<double, 9> identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        fixture.problem.cameras.assign(7, makeCamera(0.0, 0.0));
        fixture.truthRig.captures = {{0, 0, identity, {-4.0, 0.0, 0.0}, true},
                                     {0, 1, identity, {-1.4, 0.25, 0.0}, false},
                                     {0, 2, identity, {1.2, -0.2, 0.0}, false},
                                     {1, 0, identity, {3.4, 0.15, 0.0}, true},
                                     {1, 1, identity, {5.8, -0.1, 0.0}, false}};
        fixture.truthRig.sensors = {{0, 0, identity, {-0.4, 0.0, 0.0}, true},
                                    {0, 1, identity, {0.45, 0.12, 0.03}, false},
                                    {1, 0, identity, {0.0, 0.0, 0.0}, true}};
        // Rig 0, capture 1 intentionally has no sensor-1 frame. Bindings only describe images that exist.
        fixture.truthRig.cameraBindings = {
            {0, 0, 0, 0}, {1, 0, 0, 1}, {2, 0, 1, 0}, {3, 0, 2, 0}, {4, 0, 2, 1}, {5, 1, 0, 0}, {6, 1, 1, 0}};
        EXPECT_TRUE(plabundle::composeRigCameras(fixture.problem.cameras, fixture.truthRig, &fixture.truthCameras));

        fixture.problem.rig = fixture.truthRig;
        fixture.problem.rig.captures[1].rigCenterInWorld = {-1.12, 0.08, 0.1};
        fixture.problem.rig.captures[2].rigCenterInWorld = {1.0, -0.02, -0.09};
        fixture.problem.rig.captures[4].rigCenterInWorld = {5.55, 0.07, 0.08};
        fixture.problem.rig.sensors[1].cameraCenterInRig = {0.62, 0.02, 0.11};

        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 5; ++column)
            {
                const std::array<double, 3> truth{
                    {-1.6 + 0.75 * column, -1.0 + 0.65 * row, 14.0 + 0.3 * ((row + column) % 3)}};
                fixture.truthPoints.push_back(truth);
                plabundle::Track track;
                track.initialPoint = {truth[0] + 0.1, truth[1] - 0.07, truth[2] + 0.16};
                track.controlPointConstraints.push_back(
                    {truth, 0.03, 1.0, static_cast<int>(fixture.problem.tracks.size())});
                track.laserPlaneConstraints.push_back({truth, {0.0, 0.0, 1.0}, 1.0, 0.0, 0});
                for (std::size_t camera_index = 0; camera_index < fixture.truthCameras.size(); ++camera_index)
                {
                    const auto pixel = projectPoint(fixture.truthCameras[camera_index], truth);
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
                }
                fixture.problem.tracks.push_back(std::move(track));
            }
        }

        fixture.problem.scaleBarConstraints.push_back(
            {0, 1, distance(fixture.truthPoints[0], fixture.truthPoints[1]), 0.02, 1.0, 0});
        fixture.problem.cameraPosePriors.resize(fixture.truthCameras.size());
        for (const std::size_t camera_index : {2U, 6U})
        {
            plabundle::CameraPosePrior prior;
            prior.cameraCenter = fixture.truthCameras[camera_index].cameraCenter;
            prior.cameraToWorldRotation = fixture.truthCameras[camera_index].cameraToWorldRotation;
            prior.tangentFrame = plabundle::PosePriorTangentFrame::PriorCamera;
            if (camera_index == 2U)
            {
                prior.uncertainty = plabundle::PosePriorUncertainty::Covariance;
                for (int axis = 0; axis < 3; ++axis)
                {
                    prior.uncertaintyMatrix[static_cast<std::size_t>(axis * 6 + axis)] = 0.01;
                    prior.uncertaintyMatrix[static_cast<std::size_t>((axis + 3) * 6 + axis + 3)] = 0.04;
                }
                prior.uncertaintyMatrix[3] = 0.002;
                prior.uncertaintyMatrix[18] = 0.002;
            }
            else
            {
                prior.uncertainty = plabundle::PosePriorUncertainty::SqrtInformation;
                for (int axis = 0; axis < 3; ++axis)
                {
                    prior.uncertaintyMatrix[static_cast<std::size_t>(axis * 6 + axis)] = 10.0;
                    prior.uncertaintyMatrix[static_cast<std::size_t>((axis + 3) * 6 + axis + 3)] = 5.0;
                }
            }
            fixture.problem.cameraPosePriors[camera_index] = prior;
        }

        fixture.problem.cameraPlaneConstraint =
            plabundle::CameraPlaneConstraint{{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {}, 0.1, 5.0};
        const std::array<double, 3> laser_truth{{0.35, -0.25, 11.8}};
        plabundle::LaserRangeConstraint shot;
        shot.cameraIndex = 4;
        shot.initialPoint = {laser_truth[0] + 0.2, laser_truth[1] - 0.15, laser_truth[2] + 0.25};
        shot.sigmaRangeMeters = 0.03;
        shot.leverArmCameraMeters = {0.12, -0.04, 0.06};
        const auto& source = fixture.truthCameras[4];
        const std::array<double, 3> emitter{{source.cameraCenter[0] + shot.leverArmCameraMeters[0],
                                             source.cameraCenter[1] + shot.leverArmCameraMeters[1],
                                             source.cameraCenter[2] + shot.leverArmCameraMeters[2]}};
        shot.observedRangeMeters = distance(laser_truth, emitter);
        shot.pointMode = plabundle::LaserPointMode::Constrained;
        shot.pointPrior = laser_truth;
        shot.pointPriorSqrtInformation = {5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0};
        shot.shotId = "rig-survey-shot";
        for (const int camera_index : {0, 5})
        {
            const auto pixel = projectPoint(fixture.truthCameras[static_cast<std::size_t>(camera_index)], laser_truth);
            shot.measuredImageObservations.push_back({camera_index, pixel[0], pixel[1], 1.0, 1.0});
        }
        fixture.problem.laserRangeConstraints.push_back(std::move(shot));
        fixture.problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
        return fixture;
    }

    plabundle::Options makeRigSurveyOptions(plabundle::Backend backend)
    {
        plabundle::Options options;
        options.backend = backend;
        options.refineCameraPose = true;
        options.controlPointWeight = 40.0;
        options.controlPointHuberDeltaMeters = 0.0;
        options.laserPlaneWeight = 20.0;
        options.laserHuberDeltaMeters = 0.0;
        options.scaleBarWeight = 50.0;
        options.scaleBarHuberDeltaMeters = 0.0;
        options.cameraPosePriorWeight = 15.0;
        options.cameraPosePriorHuberDelta = 0.0;
        options.cameraPlaneHuberDelta = 0.0;
        options.laserRangeWeight = 10.0;
        options.laserRangeHuberDelta = 0.0;
        options.enablePointFilter = false;
        options.maxIterations = 50;
        options.allowBackendFallback = false;
        return options;
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
        const double rotation_sigma = prior.rotationSigmaDegrees * 3.14159265358979323846 / 180.0;
        if (camera_index == 2U)
        {
            prior.uncertainty = plabundle::PosePriorUncertainty::Covariance;
            for (int axis = 0; axis < 3; ++axis)
            {
                prior.uncertaintyMatrix[static_cast<std::size_t>(axis * 6 + axis)] = rotation_sigma * rotation_sigma;
                prior.uncertaintyMatrix[static_cast<std::size_t>((axis + 3) * 6 + axis + 3)] =
                    prior.positionSigmaMeters * prior.positionSigmaMeters;
            }
            prior.uncertaintyMatrix[3] = 2.0e-5;
            prior.uncertaintyMatrix[18] = 2.0e-5;
        }
        else
        {
            prior.uncertainty = plabundle::PosePriorUncertainty::SqrtInformation;
            for (int axis = 0; axis < 3; ++axis)
            {
                prior.uncertaintyMatrix[static_cast<std::size_t>(axis * 6 + axis)] = 1.0 / rotation_sigma;
                prior.uncertaintyMatrix[static_cast<std::size_t>((axis + 3) * 6 + axis + 3)] =
                    1.0 / prior.positionSigmaMeters;
            }
            prior.uncertaintyMatrix[4] = 0.05;
        }
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

TEST(PlaBundleSolverTest, RefinesRigCaptureAndSharedSensorExtrinsicWithMissingSensorCapture)
{
    plabundle::RigTopology truth_rig;
    plabundle::Problem problem = makeRigProblem(&truth_rig);
    const double capture_error_before =
        distance(problem.rig.captures[1].rigCenterInWorld, truth_rig.captures[1].rigCenterInWorld);
    const double sensor_error_before =
        distance(problem.rig.sensors[1].cameraCenterInRig, truth_rig.sensors[1].cameraCenterInRig);

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 30;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.refinedRig.captures.size(), truth_rig.captures.size());
    ASSERT_EQ(result.refinedRig.sensors.size(), truth_rig.sensors.size());
    EXPECT_LT(result.quality.meanRmsAfter, result.quality.meanRmsBefore);
    EXPECT_LT(distance(result.refinedRig.captures[1].rigCenterInWorld, truth_rig.captures[1].rigCenterInWorld),
              capture_error_before);
    EXPECT_LT(distance(result.refinedRig.sensors[1].cameraCenterInRig, truth_rig.sensors[1].cameraCenterInRig),
              sensor_error_before);
    ASSERT_EQ(result.refinedCameras.size(), problem.cameras.size());
}

TEST(PlaBundleSolverTest, SolvesComposedRigSurveyConstraintsAndMatchesAcceleratedBackends)
{
    const RigSurveyFixture fixture = makeRigSurveyFixture();
    const plabundle::Result cpu =
        plabundle::Solver().solve(fixture.problem, makeRigSurveyOptions(plabundle::Backend::PlaMatrixCpu));
    ASSERT_TRUE(cpu.usable()) << cpu.backendMessage;
    EXPECT_LT(cpu.quality.meanRmsAfter, cpu.quality.meanRmsBefore);
    EXPECT_GT(cpu.quality.controlPointConstraintCount, 0);
    EXPECT_GT(cpu.quality.laserConstraintCount, 0);
    EXPECT_EQ(cpu.quality.scaleBarConstraintCount, 1);
    EXPECT_EQ(cpu.quality.laserRangeConstraintCount, 1);
    ASSERT_EQ(cpu.refinedRig.captures.size(), fixture.truthRig.captures.size());
    ASSERT_EQ(cpu.refinedRig.sensors.size(), fixture.truthRig.sensors.size());
    EXPECT_LT(
        distance(cpu.refinedRig.captures[1].rigCenterInWorld, fixture.truthRig.captures[1].rigCenterInWorld),
        distance(fixture.problem.rig.captures[1].rigCenterInWorld, fixture.truthRig.captures[1].rigCenterInWorld));
    EXPECT_LT(
        distance(cpu.refinedRig.sensors[1].cameraCenterInRig, fixture.truthRig.sensors[1].cameraCenterInRig),
        distance(fixture.problem.rig.sensors[1].cameraCenterInRig, fixture.truthRig.sensors[1].cameraCenterInRig));
    EXPECT_EQ(cpu.refinedRig.sensors[0].cameraCenterInRig, fixture.problem.rig.sensors[0].cameraCenterInRig);
    ASSERT_EQ(cpu.laserRangeShots.size(), 1U);
    EXPECT_TRUE(cpu.laserRangeShots.front().valid);

    for (const plabundle::Backend backend :
         {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl, plabundle::Backend::PlaMatrixVulkan})
    {
        if (!plabundle::Solver::isBackendAvailable(backend))
        {
            continue;
        }
        SCOPED_TRACE(plabundle::backendName(backend));
        const plabundle::Result accelerated = plabundle::Solver().solve(fixture.problem, makeRigSurveyOptions(backend));
        ASSERT_TRUE(accelerated.usable()) << accelerated.backendMessage;
        EXPECT_EQ(accelerated.usedBackend, backend);
        EXPECT_TRUE(accelerated.usedGpu);
        EXPECT_EQ(accelerated.quality.controlPointConstraintCount, cpu.quality.controlPointConstraintCount);
        EXPECT_EQ(accelerated.quality.laserConstraintCount, cpu.quality.laserConstraintCount);
        EXPECT_EQ(accelerated.quality.scaleBarConstraintCount, cpu.quality.scaleBarConstraintCount);
        EXPECT_EQ(accelerated.quality.laserRangeConstraintCount, cpu.quality.laserRangeConstraintCount);
        EXPECT_NEAR(accelerated.quality.meanRmsAfter, cpu.quality.meanRmsAfter, 2.0e-3);
        ASSERT_EQ(accelerated.refinedCameras.size(), cpu.refinedCameras.size());
        for (std::size_t camera_index = 0; camera_index < cpu.refinedCameras.size(); ++camera_index)
        {
            EXPECT_LT(distance(accelerated.refinedCameras[camera_index].cameraCenter,
                               cpu.refinedCameras[camera_index].cameraCenter),
                      3.0e-2);
        }
        ASSERT_EQ(accelerated.laserRangeShots.size(), cpu.laserRangeShots.size());
        EXPECT_LT(distance(accelerated.laserRangeShots.front().point, cpu.laserRangeShots.front().point), 3.0e-2);
    }
}

TEST(PlaBundleSolverTest, RefinesCapturesAcrossMultipleRigs)
{
    plabundle::RigTopology truth_rig;
    plabundle::Problem problem = makeMultiRigProblem(&truth_rig);
    const double first_error_before =
        distance(problem.rig.captures[1].rigCenterInWorld, truth_rig.captures[1].rigCenterInWorld);
    const double second_error_before =
        distance(problem.rig.captures[2].rigCenterInWorld, truth_rig.captures[2].rigCenterInWorld);
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 25;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_LT(distance(result.refinedRig.captures[1].rigCenterInWorld, truth_rig.captures[1].rigCenterInWorld),
              first_error_before);
    EXPECT_LT(distance(result.refinedRig.captures[2].rigCenterInWorld, truth_rig.captures[2].rigCenterInWorld),
              second_error_before);
}

TEST(PlaBundleSolverTest, RigExplicitGaugeRequiresFixedCaptureGeometry)
{
    plabundle::RigTopology truth_rig;
    plabundle::Problem problem = makeMultiRigProblem(&truth_rig);
    for (plabundle::RigCapture& capture : problem.rig.captures)
    {
        capture.fixedPose = false;
    }
    problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    const plabundle::Result rejected = plabundle::Solver().solve(problem, options);
    EXPECT_EQ(rejected.status, plabundle::SolveStatus::UnsupportedConfiguration);
    EXPECT_FALSE(rejected.usable());

    problem.rig.captures[0].fixedPose = true;
    problem.rig.captures[3].fixedPose = true;
    const plabundle::Result accepted = plabundle::Solver().solve(problem, options);
    EXPECT_TRUE(accepted.usable()) << accepted.backendMessage;
}

TEST(PlaBundleSolverTest, RigPointOnlySolveKeepsTopologyFixed)
{
    plabundle::RigTopology truth_rig;
    const plabundle::Problem problem = makeRigProblem(&truth_rig);
    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.refinedRig.captures.size(), problem.rig.captures.size());
    ASSERT_EQ(result.refinedRig.sensors.size(), problem.rig.sensors.size());
    EXPECT_EQ(result.refinedRig.captures[1].rigCenterInWorld, problem.rig.captures[1].rigCenterInWorld);
    EXPECT_EQ(result.refinedRig.sensors[1].cameraCenterInRig, problem.rig.sensors[1].cameraCenterInRig);
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

TEST(PlaBundleSolverTest, VulkanSolvesDoubleBundleProblemThroughMixedPrecisionPath)
{
    if (!plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixVulkan))
    {
        GTEST_SKIP() << "PlaMatrix Vulkan backend is unavailable";
    }

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixVulkan;
    options.refineCameraPose = true;
    options.enablePointFilter = false;
    options.maxIterations = 40;
    options.allowBackendFallback = false;
    const plabundle::Result result = plabundle::Solver().solve(makeJointPoseProblem(), options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_EQ(result.requestedBackend, plabundle::Backend::PlaMatrixVulkan);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixVulkan);
    EXPECT_TRUE(result.usedGpu);
    EXPECT_TRUE(result.plaMatrix.mixedPrecisionUsed);
    EXPECT_FALSE(result.plaMatrix.schurAssemblyOnDevice);
    EXPECT_FALSE(result.plaMatrix.deviceName.empty());
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

TEST(PlaBundleSolverTest, RecoversAndWritesBackSharedMetashapeExtensionParameters)
{
    const MetashapeExtensionFixture fixture = makeMetashapeExtensionFixture();
    const plabundle::Result result =
        plabundle::Solver().solve(fixture.problem, makeMetashapeExtensionOptions(plabundle::Backend::PlaMatrixCpu));
    ASSERT_TRUE(result.usable()) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), fixture.problem.cameras.size());
    EXPECT_LT(result.quality.meanRmsAfter, result.quality.meanRmsBefore * 0.05);
    plabundle::MetashapeFrameCalibration refined;
    ASSERT_TRUE(plabundle::metashapeFrameCalibration(result.refinedCameras.front(), &refined));
    EXPECT_NEAR(refined.b2, fixture.truthCalibration.b2, 0.5);
    EXPECT_NEAR(refined.k4, fixture.truthCalibration.k4, 0.01);
    EXPECT_NEAR(refined.p3, fixture.truthCalibration.p3, 0.03);
    EXPECT_NEAR(refined.p4, fixture.truthCalibration.p4, 0.04);
    EXPECT_NEAR(result.quality.refinedSharedSkewB2, refined.b2, 1.0e-8);
    EXPECT_NEAR(result.quality.refinedSharedRadialK4, refined.k4, 1.0e-10);
    EXPECT_NEAR(result.quality.refinedSharedTangentialP3, refined.p3, 1.0e-10);
    EXPECT_NEAR(result.quality.refinedSharedTangentialP4, refined.p4, 1.0e-10);
    for (const plabundle::FrameCamera& camera : result.refinedCameras)
    {
        EXPECT_EQ(camera.brownTangentialConvention, plabundle::BrownTangentialConvention::Metashape);
        EXPECT_NEAR(camera.skewPixels, refined.b2, 1.0e-8);
        EXPECT_NEAR(camera.distortion.k4, refined.k4, 1.0e-10);
        EXPECT_NEAR(camera.distortion.p3, refined.p3, 1.0e-10);
        EXPECT_NEAR(camera.distortion.p4, refined.p4, 1.0e-10);
    }
}

TEST(PlaBundleSolverTest, AcceleratedBackendsMatchCpuForCompleteMetashapeCalibration)
{
    const bool cuda_available = plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCuda);
    const bool opencl_available = plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixOpenCl);
    if (!cuda_available && !opencl_available)
    {
        GTEST_SKIP() << "CUDA and OpenCL backends are unavailable";
    }
    const MetashapeExtensionFixture fixture = makeMetashapeExtensionFixture();
    const plabundle::Result cpu =
        plabundle::Solver().solve(fixture.problem, makeMetashapeExtensionOptions(plabundle::Backend::PlaMatrixCpu));
    ASSERT_TRUE(cpu.usable()) << cpu.backendMessage;
    plabundle::MetashapeFrameCalibration cpu_calibration;
    ASSERT_TRUE(plabundle::metashapeFrameCalibration(cpu.refinedCameras.front(), &cpu_calibration));

    for (const plabundle::Backend backend : {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl})
    {
        if (!plabundle::Solver::isBackendAvailable(backend))
        {
            continue;
        }
        SCOPED_TRACE(plabundle::backendName(backend));
        const plabundle::Result accelerated =
            plabundle::Solver().solve(fixture.problem, makeMetashapeExtensionOptions(backend));
        ASSERT_TRUE(accelerated.usable()) << accelerated.backendMessage;
        EXPECT_EQ(accelerated.usedBackend, backend);
        EXPECT_TRUE(accelerated.usedGpu);
        plabundle::MetashapeFrameCalibration calibration;
        ASSERT_TRUE(plabundle::metashapeFrameCalibration(accelerated.refinedCameras.front(), &calibration));
        EXPECT_NEAR(accelerated.quality.meanRmsAfter, cpu.quality.meanRmsAfter, 2.0e-3);
        EXPECT_NEAR(calibration.b2, cpu_calibration.b2, 0.1);
        EXPECT_NEAR(calibration.k4, cpu_calibration.k4, 2.0e-3);
        EXPECT_NEAR(calibration.p3, cpu_calibration.p3, 5.0e-3);
        EXPECT_NEAR(calibration.p4, cpu_calibration.p4, 5.0e-3);
    }
}

TEST(PlaBundleSolverTest, SupportsFixedSharedAndIndependentNonBrownCalibrationGroups)
{
    const auto set_model = [](plabundle::FrameCamera* camera,
                              plabundle::FrameProjectionModel model,
                              const std::array<double, 5>& coefficients)
    {
        camera->projectionModel = model;
        camera->modelCoefficients = coefficients;
    };

    std::vector<plabundle::FrameCamera> spherical_cameras{
        makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -2.5), makeCamera(0.0, 2.5)};
    for (auto& camera : spherical_cameras)
    {
        set_model(&camera, plabundle::FrameProjectionModel::Spherical, {});
    }
    const plabundle::Problem fixed_problem = makeFullBrownProblem(spherical_cameras, spherical_cameras);
    plabundle::Options fixed_options;
    fixed_options.backend = plabundle::Backend::PlaMatrixCpu;
    fixed_options.refineCameraPose = false;
    fixed_options.enablePointFilter = false;
    const plabundle::Result fixed = plabundle::Solver().solve(fixed_problem, fixed_options);
    ASSERT_TRUE(fixed.usable()) << fixed.backendMessage;
    EXPECT_EQ(fixed.refinedCameras.front().projectionModel, plabundle::FrameProjectionModel::Spherical);
    EXPECT_EQ(plabundle::cameraParameterBlock(fixed.refinedCameras.front()),
              plabundle::cameraParameterBlock(spherical_cameras.front()));

    std::vector<plabundle::FrameCamera> fisheye_truth{makeCamera(-3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(3.0, 0.0, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, -2.5, 1040.0, 998.4, 516.0, 381.5),
                                                      makeCamera(0.0, 2.5, 1040.0, 998.4, 516.0, 381.5)};
    std::vector<plabundle::FrameCamera> fisheye_initial{makeCamera(-3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                        makeCamera(3.0, 0.0, 900.0, 900.0, 512.0, 384.0),
                                                        makeCamera(0.0, -2.5, 900.0, 900.0, 512.0, 384.0),
                                                        makeCamera(0.0, 2.5, 900.0, 900.0, 512.0, 384.0)};
    for (auto& camera : fisheye_truth)
    {
        set_model(&camera, plabundle::FrameProjectionModel::Fisheye, {{-0.025, 0.003, -0.0003, 0.00004, -0.000005}});
    }
    for (auto& camera : fisheye_initial)
    {
        set_model(&camera, plabundle::FrameProjectionModel::Fisheye, {});
    }
    const plabundle::Problem shared_problem = makeFullBrownProblem(fisheye_truth, fisheye_initial);
    plabundle::Options shared_options;
    shared_options.backend = plabundle::Backend::PlaMatrixCpu;
    shared_options.refineCameraPose = false;
    shared_options.refineSharedFocalLength = true;
    shared_options.refineSharedFocalAspectRatio = true;
    shared_options.refineSharedPrincipalPoint = true;
    shared_options.refineSharedModelCoefficients = true;
    shared_options.refineSharedHighOrderDistortion = true;
    shared_options.controlPointWeight = 10000.0;
    shared_options.controlPointHuberDeltaMeters = 0.0;
    shared_options.sharedFocalPriorSigma = 5.0;
    shared_options.sharedFocalAspectPriorSigma = 5.0;
    shared_options.sharedPrincipalPointPriorSigmaFraction = 1.0;
    shared_options.sharedRadialK1PriorSigma = 5.0;
    shared_options.sharedRadialK2PriorSigma = 5.0;
    shared_options.sharedRadialK3PriorSigma = 5.0;
    shared_options.sharedTangentialP1PriorSigma = 5.0;
    shared_options.sharedTangentialP2PriorSigma = 5.0;
    shared_options.enablePointFilter = false;
    shared_options.maxIterations = 60;
    const plabundle::Result shared = plabundle::Solver().solve(shared_problem, shared_options);
    ASSERT_TRUE(shared.usable()) << shared.backendMessage;
    EXPECT_LT(shared.quality.meanRmsAfter, shared.quality.meanRmsBefore);
    EXPECT_EQ(shared.quality.refinedCalibrationGroupCount, 1);
    EXPECT_EQ(shared.refinedCameras.front().projectionModel, plabundle::FrameProjectionModel::Fisheye);
    EXPECT_NEAR(shared.refinedCameras.front().focalXPixels, 1040.0, 35.0);
    EXPECT_NEAR(shared.refinedCameras.front().modelCoefficients[0], -0.025, 0.025);

    std::vector<plabundle::FrameCamera> equidistant_truth{makeCamera(-3.0, 0.0, 900.0, 900.0),
                                                          makeCamera(3.0, 0.0, 960.0, 960.0),
                                                          makeCamera(0.0, -2.5, 1060.0, 1060.0),
                                                          makeCamera(0.0, 2.5, 1120.0, 1120.0)};
    std::vector<plabundle::FrameCamera> equidistant_initial{makeCamera(-3.0, 0.0, 1000.0, 1000.0),
                                                            makeCamera(3.0, 0.0, 1000.0, 1000.0),
                                                            makeCamera(0.0, -2.5, 1000.0, 1000.0),
                                                            makeCamera(0.0, 2.5, 1000.0, 1000.0)};
    for (auto& camera : equidistant_truth)
    {
        set_model(&camera, plabundle::FrameProjectionModel::Equidistant, {{-0.01, 0.001, 0.0, 0.0, 0.0}});
    }
    for (auto& camera : equidistant_initial)
    {
        set_model(&camera, plabundle::FrameProjectionModel::Equidistant, {{-0.01, 0.001, 0.0, 0.0, 0.0}});
    }
    plabundle::Problem independent_problem = makeFullBrownProblem(equidistant_truth, equidistant_initial);
    independent_problem.cameraCalibrationGroupIds = {0, 1, 2, 3};
    plabundle::Options independent_options;
    independent_options.backend = plabundle::Backend::PlaMatrixCpu;
    independent_options.refineCameraPose = false;
    independent_options.refineSharedFocalLength = true;
    independent_options.controlPointWeight = 10000.0;
    independent_options.controlPointHuberDeltaMeters = 0.0;
    independent_options.sharedFocalPriorSigma = 5.0;
    independent_options.enablePointFilter = false;
    independent_options.maxIterations = 50;
    const plabundle::Result independent = plabundle::Solver().solve(independent_problem, independent_options);
    ASSERT_TRUE(independent.usable()) << independent.backendMessage;
    EXPECT_EQ(independent.quality.refinedCalibrationGroupCount, 4);
    for (std::size_t index = 0; index < equidistant_truth.size(); ++index)
    {
        EXPECT_NEAR(independent.refinedCameras[index].focalXPixels, equidistant_truth[index].focalXPixels, 30.0);
    }
}

TEST(PlaBundleSolverTest, SolvesRollingShutterPoseAtObservedRows)
{
    std::vector<plabundle::FrameCamera> truth_cameras{
        makeCamera(-3.0, 0.0), makeCamera(3.0, 0.0), makeCamera(0.0, -2.5), makeCamera(0.0, 2.5)};
    for (std::size_t index = 0; index < truth_cameras.size(); ++index)
    {
        auto& camera = truth_cameras[index];
        camera.projectionModel = plabundle::FrameProjectionModel::RollingShutter;
        camera.rollingShutter.referenceLinePixels = camera.principalYPixel;
        camera.rollingShutter.secondsPerLine = 2.0e-5;
        camera.rollingShutter.linearVelocityWorldMetersPerSecond = {
            {0.3 * static_cast<double>(index + 1), -0.15, 0.05}};
        camera.rollingShutter.angularVelocityCameraRadiansPerSecond = {{0.015, -0.01, 0.008}};
    }
    plabundle::Problem problem;
    problem.cameras = truth_cameras;
    ASSERT_TRUE(plabundle::applyPoseDelta(&problem.cameras[2], {0.004, -0.006, 0.003, 0.08, -0.05, 0.04}));
    ASSERT_TRUE(plabundle::applyPoseDelta(&problem.cameras[3], {-0.003, 0.005, -0.002, -0.07, 0.04, -0.03}));
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            const std::array<double, 3> truth{
                {-1.4 + 0.7 * column, -1.0 + 0.65 * row, 12.0 + 0.2 * ((row + column) % 3)}};
            plabundle::Track track;
            track.initialPoint = truth;
            track.controlPointConstraints.push_back({truth, 0.01, 1.0, static_cast<int>(problem.tracks.size())});
            for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
            {
                const auto pixel = projectRollingPoint(truth_cameras[camera_index], truth);
                track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
            }
            problem.tracks.push_back(std::move(track));
        }
    }
    problem.fixedCameraIndices = {0, 1};
    problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = true;
    options.controlPointWeight = 10000.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.enablePointFilter = false;
    options.maxIterations = 40;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_LT(result.quality.meanRmsAfter, result.quality.meanRmsBefore);
    EXPECT_EQ(result.refinedCameras[2].projectionModel, plabundle::FrameProjectionModel::RollingShutter);
    for (std::size_t index : {2U, 3U})
    {
        const double initial_error = distance(problem.cameras[index].cameraCenter, truth_cameras[index].cameraCenter);
        const double refined_error =
            distance(result.refinedCameras[index].cameraCenter, truth_cameras[index].cameraCenter);
        EXPECT_LT(refined_error, 0.7 * initial_error);
    }
}

TEST(PlaBundleSolverTest, RejectsMixedProjectionModelsInsideSharedCalibrationGroup)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-1.0, 0.0), makeCamera(1.0, 0.0)};
    problem.cameras[1].projectionModel = plabundle::FrameProjectionModel::Equidistant;
    plabundle::Track track;
    track.initialPoint = {{0.0, 0.0, 8.0}};
    for (std::size_t index = 0; index < problem.cameras.size(); ++index)
    {
        const auto pixel = projectPoint(problem.cameras[index], track.initialPoint);
        track.observations.push_back({static_cast<int>(index), pixel[0], pixel[1], 1.0, 1.0});
    }
    problem.tracks.push_back(track);
    plabundle::Options options;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::InvalidInput);
    EXPECT_FALSE(result.usable());
}

TEST(PlaBundleSolverTest, RejectsMixedTangentialConventionsInsideSharedCalibrationGroup)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-1.0, 0.0), makeCamera(1.0, 0.0)};
    ASSERT_TRUE(plabundle::applyMetashapeFrameCalibration(
        &problem.cameras[1], {950.0, 512.0, 384.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    plabundle::Track track;
    track.initialPoint = {{0.0, 0.0, 8.0}};
    for (std::size_t index = 0; index < problem.cameras.size(); ++index)
    {
        const auto pixel = projectPoint(problem.cameras[index], track.initialPoint);
        track.observations.push_back({static_cast<int>(index), pixel[0], pixel[1], 1.0, 1.0});
    }
    problem.tracks.push_back(track);
    plabundle::Options options;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);
    EXPECT_EQ(result.status, plabundle::SolveStatus::InvalidInput);
    EXPECT_FALSE(result.usable());
    EXPECT_NE(result.backendMessage.find("约定"), std::string::npos);
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

TEST(PlaBundleSolverTest, ReferenceOnlineSchurMatchesGeneralSchurPathWithCauchyAndControlCovariance)
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
        ASSERT_EQ(track.controlPointConstraints.size(), 1U);
        auto& control = track.controlPointConstraints.front();
        control.uncertainty = plabundle::ControlPointUncertainty::Covariance;
        control.uncertaintyMatrix = {1.0e-4, 2.0e-5, 0.0, 2.0e-5, 2.0e-4, 0.0, 0.0, 0.0, 4.0e-4};
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
    options.controlPointHuberDeltaMeters = 0.0;
    options.enablePointFilter = false;
    options.imageRobustLoss = plabundle::ImageRobustLoss::Cauchy;
    options.imageRobustLossScalePixels = 5.0;
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

TEST(PlaBundleSolverTest, CauchyImageLossReducesGrossObservationBias)
{
    const plabundle::Problem problem = makeFilteredTrackProblem();
    const std::array<double, 3> truth{{1.0, -0.09, 8.25}};

    plabundle::Options least_squares_options;
    least_squares_options.backend = plabundle::Backend::PlaMatrixCpu;
    least_squares_options.refineCameraPose = false;
    least_squares_options.enablePointFilter = false;
    least_squares_options.enableBackendQualityGate = false;
    least_squares_options.maxIterations = 30;
    const plabundle::Result least_squares = plabundle::Solver().solve(problem, least_squares_options);

    plabundle::Options cauchy_options = least_squares_options;
    cauchy_options.imageRobustLoss = plabundle::ImageRobustLoss::Cauchy;
    cauchy_options.imageRobustLossScalePixels = 2.0;
    const plabundle::Result cauchy = plabundle::Solver().solve(problem, cauchy_options);

    ASSERT_TRUE(least_squares.usable()) << least_squares.backendMessage;
    ASSERT_TRUE(cauchy.usable()) << cauchy.backendMessage;
    ASSERT_EQ(least_squares.points.size(), problem.tracks.size());
    ASSERT_EQ(cauchy.points.size(), problem.tracks.size());
    const double least_squares_error = distance(least_squares.points.back().point, truth);
    const double cauchy_error = distance(cauchy.points.back().point, truth);
    EXPECT_LT(cauchy_error, least_squares_error * 0.25);
    EXPECT_LT(cauchy_error, 0.05);
}

TEST(PlaBundleSolverTest, ControlPointQualityRemainsUnwhitenedMetricRms)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 8.0}};
    plabundle::Track track;
    track.initialPoint = {truth[0] + 0.3, truth[1] - 0.4, truth[2] + 0.5};
    for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
    {
        const auto pixel = projectPoint(problem.cameras[camera_index], truth);
        track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
    }
    plabundle::ControlPointConstraint control;
    control.point = truth;
    control.uncertainty = plabundle::ControlPointUncertainty::Covariance;
    control.uncertaintyMatrix = {0.01, 0.004, 0.0, 0.004, 0.04, 0.0, 0.0, 0.0, 0.25};
    track.controlPointConstraints.push_back(control);
    problem.tracks.push_back(track);

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableBackendQualityGate = false;
    options.controlPointHuberDeltaMeters = 0.0;
    options.maxIterations = 20;
    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_NEAR(result.quality.controlPointRmsBeforeMeters, std::sqrt(0.5), 1.0e-12);
    EXPECT_LT(result.quality.controlPointRmsAfterMeters, 1.0e-4);
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

TEST(PlaBundleSolverTest, InvalidTrialProjectionBacktracksWithoutAbortingSolve)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-1.0, 0.0), makeCamera(1.0, 0.0)};
    plabundle::Track track;
    track.initialPoint = {0.0, 0.0, 1.0};
    for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
    {
        const auto pixel = projectPoint(problem.cameras[camera_index], track.initialPoint);
        track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
    }
    // The first Gauss-Newton trial follows this dominant control residual behind both cameras.
    // A reduced Armijo step remains in the valid projection domain and must be considered.
    track.controlPointConstraints.push_back({{0.0, 0.0, -10.0}, 1.0e-6, 1.0, 0});
    problem.tracks.push_back(std::move(track));

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableBackendQualityGate = false;
    options.controlPointWeight = 1.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.maxIterations = 12;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    EXPECT_NE(result.status, plabundle::SolveStatus::NumericalFailure) << result.backendMessage;
    EXPECT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_GT(result.plaMatrix.acceptedSteps, 0);
    EXPECT_NE(result.backendMessage.find("Armijo"), std::string::npos);
    EXPECT_EQ(result.backendMessage.find("非法投影"), std::string::npos);
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
    EXPECT_NE(result.backendMessage.find("参考联合 BA"), std::string::npos);
    EXPECT_NE(result.backendMessage.find("valid-track ratio"), std::string::npos);
    EXPECT_EQ(result.quality.optimizedTracks, 5);
}

TEST(PlaBundleSolverTest, ExplicitCpuQualityGateRejectsReprojectionDegradation)
{
    const plabundle::Problem problem = makeQualityDegradingConstraintProblem();
    const plabundle::Options options = makeQualityDegradingConstraintOptions(plabundle::Backend::PlaMatrixCpu);

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    EXPECT_EQ(result.requestedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(result.status, plabundle::SolveStatus::NumericalFailure);
    EXPECT_FALSE(result.usable());
    EXPECT_TRUE(result.qualityGateRejected);
    EXPECT_GT(result.quality.meanRmsAfter, result.quality.meanRmsBefore + 1.0e-6);
    EXPECT_NE(result.qualityGateMessage.find("reprojection RMS growth"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("explicit_backend"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("quality_gate_rejected"), std::string::npos);
}

TEST(PlaBundleSolverTest, QualityGateAllowsZeroInitialControlResidualWithinMeasurementUncertainty)
{
    plabundle::Problem problem;
    problem.cameras = {makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> control_point{{0.0, 0.0, 10.0}};
    const std::array<double, 3> image_point{{0.08, -0.04, 10.0}};
    plabundle::Track track;
    track.initialPoint = control_point;
    track.controlPointConstraints.push_back({control_point, 0.1, 1.0, 0});
    for (std::size_t camera_index = 0; camera_index < problem.cameras.size(); ++camera_index)
    {
        const auto pixel = projectPoint(problem.cameras[camera_index], image_point);
        track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0, 1.0});
    }
    problem.tracks.push_back(std::move(track));

    plabundle::Options options;
    options.backend = plabundle::Backend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.controlPointHuberDeltaMeters = 0.0;
    options.maxIterations = 20;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_FALSE(result.qualityGateRejected);
    EXPECT_DOUBLE_EQ(result.quality.controlPointRmsBeforeMeters, 0.0);
    EXPECT_GT(result.quality.controlPointRmsAfterMeters, 1.0e-6);
    EXPECT_LT(result.quality.controlPointRmsAfterMeters, std::sqrt(3.0) * 0.1);
    EXPECT_LT(result.quality.meanRmsAfter, result.quality.meanRmsBefore);
}

TEST(PlaBundleSolverTest, ExplicitAcceleratedQualityGateRejectsDegradationAndChecksCpuFallback)
{
    const plabundle::Problem problem = makeQualityDegradingConstraintProblem();
    bool exercised_backend = false;
    for (const plabundle::Backend backend :
         {plabundle::Backend::PlaMatrixCuda, plabundle::Backend::PlaMatrixOpenCl, plabundle::Backend::PlaMatrixVulkan})
    {
        if (!plabundle::Solver::isBackendAvailable(backend))
        {
            continue;
        }
        exercised_backend = true;
        SCOPED_TRACE(plabundle::backendName(backend));
        plabundle::Options options = makeQualityDegradingConstraintOptions(backend);

        const plabundle::Result rejected = plabundle::Solver().solve(problem, options);

        EXPECT_EQ(rejected.requestedBackend, backend);
        EXPECT_EQ(rejected.usedBackend, backend);
        EXPECT_FALSE(rejected.usable());
        EXPECT_TRUE(rejected.qualityGateRejected);
        EXPECT_FALSE(rejected.backendFallback);
        EXPECT_NE(rejected.qualityGateMessage.find("reprojection RMS growth"), std::string::npos);

        options.allowBackendFallback = true;
        const plabundle::Result fallback = plabundle::Solver().solve(problem, options);

        EXPECT_EQ(fallback.requestedBackend, backend);
        EXPECT_EQ(fallback.usedBackend, plabundle::Backend::PlaMatrixCpu);
        EXPECT_FALSE(fallback.usable());
        EXPECT_TRUE(fallback.backendFallback);
        EXPECT_TRUE(fallback.qualityGateRejected);
        EXPECT_NE(fallback.qualityGateMessage.find("candidate quality rejection"), std::string::npos);
        EXPECT_NE(fallback.qualityGateMessage.find("CPU fallback quality rejection"), std::string::npos);
        EXPECT_NE(fallback.backendSelectionReason.find("accelerated_candidate_quality_gate_rejected"),
                  std::string::npos);
        EXPECT_NE(fallback.backendSelectionReason.find("cpu_fallback_quality_gate_rejected"), std::string::npos);
    }
    if (!exercised_backend)
    {
        GTEST_SKIP() << "no accelerated PlaMatrix backend is available";
    }
}

TEST(PlaBundleSolverTest, AutoQualityGateRejectsWhenCpuFallbackAlsoFails)
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

    EXPECT_FALSE(result.usable()) << result.backendMessage;
    EXPECT_EQ(result.status, plabundle::SolveStatus::NumericalFailure);
    EXPECT_EQ(result.requestedBackend, plabundle::Backend::Auto);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_FALSE(result.usedGpu);
    EXPECT_TRUE(result.qualityGateRejected);
    EXPECT_NE(result.qualityGateMessage.find("candidate quality rejection"), std::string::npos);
    EXPECT_NE(result.qualityGateMessage.find("CPU fallback quality rejection"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("accelerated_candidate_quality_gate_rejected"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("fallback_to_plamatrix_cpu"), std::string::npos);
    EXPECT_NE(result.backendSelectionReason.find("cpu_fallback_quality_gate_rejected"), std::string::npos);
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
