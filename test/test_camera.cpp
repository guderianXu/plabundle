#include <plabundle/camera.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>

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
