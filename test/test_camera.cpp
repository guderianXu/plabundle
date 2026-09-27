#include <plabundle/camera.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

    plabundle::FrameCamera makeCamera()
    {
        plabundle::FrameCamera camera;
        camera.focalXPixels = 1200.0;
        camera.focalYPixels = 1180.0;
        camera.principalXPixel = 640.0;
        camera.principalYPixel = 480.0;
        camera.pixelPitchMillimeters = 0.0045;
        camera.imageSize = plabundle::ImageSize{1280, 960};
        return camera;
    }

} // namespace

TEST(PlaBundleCameraTest, ProjectsBrownConradyWithSignedPixelAxes)
{
    plabundle::FrameCamera camera = makeCamera();
    camera.uAxisSign = -1;
    camera.distortion = {0.01, -0.002, 0.0003, 0.0004, -0.0005};
    const std::array<double, 3> world{0.4, -0.2, 2.0};

    plabundle::Projection projection;
    ASSERT_TRUE(plabundle::projectWorldPoint(camera, world, &projection));

    const double x = world[0] / world[2];
    const double y = world[1] / world[2];
    const double r2 = x * x + y * y;
    const double radial =
        1.0 + camera.distortion.k1 * r2 + camera.distortion.k2 * r2 * r2 + camera.distortion.k3 * r2 * r2 * r2;
    const double xd = x * radial + 2.0 * camera.distortion.p1 * x * y + camera.distortion.p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + camera.distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * camera.distortion.p2 * x * y;
    EXPECT_NEAR(projection.pixel[0], -camera.focalXPixels * xd + camera.principalXPixel, 1.0e-12);
    EXPECT_NEAR(projection.pixel[1], camera.focalYPixels * yd + camera.principalYPixel, 1.0e-12);
    EXPECT_DOUBLE_EQ(projection.positiveDepth, 2.0);
}

TEST(PlaBundleCameraTest, HonorsFlippedPositiveDepth)
{
    plabundle::FrameCamera camera = makeCamera();
    camera.depthAxisFlipped = true;

    plabundle::Projection projection;
    EXPECT_FALSE(plabundle::projectWorldPoint(camera, {0.0, 0.0, 2.0}, &projection));
    ASSERT_TRUE(plabundle::projectWorldPoint(camera, {0.0, 0.0, -2.0}, &projection));
    EXPECT_DOUBLE_EQ(projection.positiveDepth, 2.0);
    EXPECT_DOUBLE_EQ(projection.pixel[0], camera.principalXPixel);
    EXPECT_DOUBLE_EQ(projection.pixel[1], camera.principalYPixel);
}

TEST(PlaBundleCameraTest, RejectsInvalidNumericalState)
{
    plabundle::FrameCamera camera = makeCamera();
    std::string error;
    EXPECT_TRUE(plabundle::validateFrameCamera(camera, &error));

    camera.pixelPitchMillimeters = 0.0;
    EXPECT_FALSE(plabundle::validateFrameCamera(camera, &error));
    EXPECT_FALSE(error.empty());

    camera = makeCamera();
    camera.uAxisSign = 0;
    EXPECT_FALSE(plabundle::validateFrameCamera(camera, &error));

    camera = makeCamera();
    camera.cameraToWorldRotation[0] = 2.0;
    EXPECT_FALSE(plabundle::validateFrameCamera(camera, &error));

    camera = makeCamera();
    camera.distortion.k1 = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(plabundle::validateFrameCamera(camera, &error));
}

TEST(PlaBundleCameraTest, AppliesPoseDeltaWithoutChangingIntrinsics)
{
    plabundle::FrameCamera camera = makeCamera();
    const double focal_x = camera.focalXPixels;
    constexpr double half_pi = 1.57079632679489661923;

    ASSERT_TRUE(plabundle::applyPoseDelta(&camera, {0.0, 0.0, half_pi, 1.0, -2.0, 3.0}));
    EXPECT_NEAR(camera.cameraToWorldRotation[0], 0.0, 1.0e-12);
    EXPECT_NEAR(camera.cameraToWorldRotation[1], -1.0, 1.0e-12);
    EXPECT_NEAR(camera.cameraToWorldRotation[3], 1.0, 1.0e-12);
    EXPECT_NEAR(camera.cameraToWorldRotation[4], 0.0, 1.0e-12);
    EXPECT_EQ(camera.cameraCenter, (std::array<double, 3>{1.0, -2.0, 3.0}));
    EXPECT_DOUBLE_EQ(camera.focalXPixels, focal_x);
}

TEST(PlaBundleCameraTest, UnifiedProjectionJacobiansMatchCentralDifferences)
{
    const std::vector<plabundle::FrameProjectionModel> models = {
        plabundle::FrameProjectionModel::BrownConrady,
        plabundle::FrameProjectionModel::Fisheye,
        plabundle::FrameProjectionModel::Equidistant,
        plabundle::FrameProjectionModel::Equisolid,
        plabundle::FrameProjectionModel::Spherical,
        plabundle::FrameProjectionModel::Cylindrical,
        plabundle::FrameProjectionModel::RollingShutter,
    };
    const std::array<double, 3> camera_point{{0.4, -0.25, 2.5}};
    for (const auto model : models)
    {
        SCOPED_TRACE(static_cast<int>(model));
        auto camera = makeCamera();
        camera.projectionModel = model;
        camera.distortion = {0.01, -0.002, 0.0003, 0.0004, -0.0005};
        camera.modelCoefficients = {{0.01, -0.002, 0.0003, -0.00004, 0.000005}};
        if (model == plabundle::FrameProjectionModel::RollingShutter)
        {
            camera.rollingShutter.secondsPerLine = 1.0e-4;
        }
        plabundle::ProjectionLinearization analytic;
        ASSERT_TRUE(plabundle::linearizeCameraPoint(camera, camera_point, &analytic));

        for (int axis = 0; axis < 3; ++axis)
        {
            auto plus_point = camera_point;
            auto minus_point = camera_point;
            constexpr double step = 1.0e-6;
            plus_point[axis] += step;
            minus_point[axis] -= step;
            plabundle::ProjectionLinearization plus;
            plabundle::ProjectionLinearization minus;
            ASSERT_TRUE(plabundle::linearizeCameraPoint(camera, plus_point, &plus));
            ASSERT_TRUE(plabundle::linearizeCameraPoint(camera, minus_point, &minus));
            for (int row = 0; row < 2; ++row)
            {
                const double numeric = (plus.projection.pixel[row] - minus.projection.pixel[row]) / (2.0 * step);
                EXPECT_NEAR(analytic.cameraPointJacobian[row * 3 + axis], numeric, 2.0e-5);
            }
        }

        const auto parameters = plabundle::cameraParameterBlock(camera);
        const auto supported = plabundle::supportedCameraParameters(model);
        for (std::size_t parameter = 0; parameter < parameters.size(); ++parameter)
        {
            if (!supported[parameter])
            {
                continue;
            }
            const double step = parameter < 4 ? 1.0e-6 : 1.0e-7;
            auto plus_parameters = parameters;
            auto minus_parameters = parameters;
            plus_parameters[parameter] += step;
            minus_parameters[parameter] -= step;
            plabundle::CameraParameterMask mask{};
            mask[parameter] = true;
            auto plus_camera = camera;
            auto minus_camera = camera;
            ASSERT_TRUE(plabundle::applyCameraParameterBlock(&plus_camera, plus_parameters, mask));
            ASSERT_TRUE(plabundle::applyCameraParameterBlock(&minus_camera, minus_parameters, mask));
            plabundle::ProjectionLinearization plus;
            plabundle::ProjectionLinearization minus;
            ASSERT_TRUE(plabundle::linearizeCameraPoint(plus_camera, camera_point, &plus));
            ASSERT_TRUE(plabundle::linearizeCameraPoint(minus_camera, camera_point, &minus));
            for (int row = 0; row < 2; ++row)
            {
                const double numeric = (plus.projection.pixel[row] - minus.projection.pixel[row]) / (2.0 * step);
                EXPECT_NEAR(
                    analytic.parameterJacobian[row * plabundle::kCameraParameterCount + parameter], numeric, 3.0e-4);
            }
        }
    }
}

TEST(PlaBundleCameraTest, MetashapeCalibrationRoundTripsAndProjectsAllThirteenParameters)
{
    plabundle::FrameCamera camera = makeCamera();
    const plabundle::MetashapeFrameCalibration calibration{
        1180.0, 641.5, 478.25, 20.0, 3.5, 0.01, -0.002, 0.0003, -0.00004, 0.0004, -0.0005, 0.03, -0.004};
    ASSERT_TRUE(plabundle::applyMetashapeFrameCalibration(&camera, calibration));

    plabundle::MetashapeFrameCalibration round_trip;
    ASSERT_TRUE(plabundle::metashapeFrameCalibration(camera, &round_trip));
    EXPECT_EQ(round_trip.f, calibration.f);
    EXPECT_EQ(round_trip.cx, calibration.cx);
    EXPECT_EQ(round_trip.cy, calibration.cy);
    EXPECT_EQ(round_trip.b1, calibration.b1);
    EXPECT_EQ(round_trip.b2, calibration.b2);
    EXPECT_EQ(round_trip.k1, calibration.k1);
    EXPECT_EQ(round_trip.k2, calibration.k2);
    EXPECT_EQ(round_trip.k3, calibration.k3);
    EXPECT_EQ(round_trip.k4, calibration.k4);
    EXPECT_EQ(round_trip.p1, calibration.p1);
    EXPECT_EQ(round_trip.p2, calibration.p2);
    EXPECT_EQ(round_trip.p3, calibration.p3);
    EXPECT_EQ(round_trip.p4, calibration.p4);

    const std::array<double, 3> point{{0.6, -0.3, 2.0}};
    const double x = point[0] / point[2];
    const double y = point[1] / point[2];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double radial =
        1.0 + calibration.k1 * r2 + calibration.k2 * r4 + calibration.k3 * r4 * r2 + calibration.k4 * r4 * r4;
    const double tangential_scale = 1.0 + calibration.p3 * r2 + calibration.p4 * r4;
    const double tangential_x =
        (calibration.p1 * (3.0 * x * x + y * y) + 2.0 * calibration.p2 * x * y) * tangential_scale;
    const double tangential_y =
        (2.0 * calibration.p1 * x * y + calibration.p2 * (x * x + 3.0 * y * y)) * tangential_scale;
    const double distorted_x = x * radial + tangential_x;
    const double distorted_y = y * radial + tangential_y;
    plabundle::Projection projection;
    ASSERT_TRUE(plabundle::projectWorldPoint(camera, point, &projection));
    EXPECT_NEAR(projection.pixel[0],
                (calibration.f + calibration.b1) * distorted_x + calibration.b2 * distorted_y + calibration.cx,
                1.0e-12);
    EXPECT_NEAR(projection.pixel[1], calibration.f * distorted_y + calibration.cy, 1.0e-12);
}

TEST(PlaBundleCameraTest, MetashapeThirteenParameterJacobiansMatchCentralDifferences)
{
    plabundle::FrameCamera camera = makeCamera();
    ASSERT_TRUE(plabundle::applyMetashapeFrameCalibration(
        &camera, {1180.0, 641.5, 478.25, 20.0, 3.5, 0.01, -0.002, 0.0003, -0.00004, 0.0004, -0.0005, 0.03, -0.004}));
    const std::array<double, 3> camera_point{{0.6, -0.3, 2.0}};
    plabundle::ProjectionLinearization analytic;
    ASSERT_TRUE(plabundle::linearizeCameraPoint(camera, camera_point, &analytic));
    const auto parameters = plabundle::cameraParameterBlock(camera);
    const auto supported = plabundle::supportedCameraParameters(camera);
    EXPECT_TRUE(std::all_of(supported.begin(), supported.end(), [](bool value) { return value; }));

    for (std::size_t parameter = 0; parameter < parameters.size(); ++parameter)
    {
        SCOPED_TRACE(parameter);
        const double step = parameter < 4 || parameter == 9 ? 1.0e-6 : 1.0e-7;
        auto plus_parameters = parameters;
        auto minus_parameters = parameters;
        plus_parameters[parameter] += step;
        minus_parameters[parameter] -= step;
        plabundle::CameraParameterMask mask{};
        mask[parameter] = true;
        auto plus_camera = camera;
        auto minus_camera = camera;
        ASSERT_TRUE(plabundle::applyCameraParameterBlock(&plus_camera, plus_parameters, mask));
        ASSERT_TRUE(plabundle::applyCameraParameterBlock(&minus_camera, minus_parameters, mask));
        plabundle::ProjectionLinearization plus;
        plabundle::ProjectionLinearization minus;
        ASSERT_TRUE(plabundle::linearizeCameraPoint(plus_camera, camera_point, &plus));
        ASSERT_TRUE(plabundle::linearizeCameraPoint(minus_camera, camera_point, &minus));
        for (int row = 0; row < 2; ++row)
        {
            const double numeric = (plus.projection.pixel[row] - minus.projection.pixel[row]) / (2.0 * step);
            EXPECT_NEAR(
                analytic.parameterJacobian[row * plabundle::kCameraParameterCount + parameter], numeric, 5.0e-4);
        }
    }
}

TEST(PlaBundleCameraTest, RollingShutterUsesObservedLineAndRejectsUnsupportedMask)
{
    auto rolling = makeCamera();
    rolling.projectionModel = plabundle::FrameProjectionModel::RollingShutter;
    rolling.rollingShutter.secondsPerLine = 0.1;
    rolling.rollingShutter.linearVelocityWorldMetersPerSecond[0] = 1.0;
    plabundle::Projection reference;
    plabundle::Projection later;
    ASSERT_TRUE(plabundle::projectWorldPointAtLine(rolling, {{0.0, 0.0, 10.0}}, 0.0, &reference));
    ASSERT_TRUE(plabundle::projectWorldPointAtLine(rolling, {{0.0, 0.0, 10.0}}, 10.0, &later));
    EXPECT_NEAR(reference.pixel[0], rolling.principalXPixel, 1.0e-12);
    EXPECT_NEAR(later.pixel[0], rolling.principalXPixel - rolling.focalXPixels / 10.0, 1.0e-10);

    auto spherical = makeCamera();
    spherical.projectionModel = plabundle::FrameProjectionModel::Spherical;
    auto parameters = plabundle::cameraParameterBlock(spherical);
    parameters[4] = 0.1;
    plabundle::CameraParameterMask mask{};
    mask[4] = true;
    std::string error;
    EXPECT_FALSE(plabundle::applyCameraParameterBlock(&spherical, parameters, mask, &error));
    EXPECT_FALSE(error.empty());
}
