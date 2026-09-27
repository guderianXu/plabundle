#include <plabundle/adaptive_camera_model.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace
{
    using Vec3 = std::array<double, 3>;

    Vec3 subtract(const Vec3& left, const Vec3& right)
    {
        return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
    }

    Vec3 cross(const Vec3& left, const Vec3& right)
    {
        return {left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0]};
    }

    Vec3 normalize(Vec3 value)
    {
        const double length = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        for (double& component : value)
        {
            component /= length;
        }
        return value;
    }

    plabundle::FrameCamera makeLookAtCamera(const Vec3& center, const Vec3& target, double focalLength = 800.0)
    {
        const Vec3 forward = normalize(subtract(target, center));
        const Vec3 reference = std::abs(forward[2]) < 0.90 ? Vec3{0.0, 0.0, 1.0} : Vec3{0.0, 1.0, 0.0};
        const Vec3 right = normalize(cross(reference, forward));
        const Vec3 up = normalize(cross(forward, right));

        plabundle::FrameCamera camera;
        camera.cameraCenter = center;
        camera.cameraToWorldRotation = {
            right[0], up[0], forward[0], right[1], up[1], forward[1], right[2], up[2], forward[2]};
        camera.focalXPixels = focalLength;
        camera.focalYPixels = focalLength;
        camera.principalXPixel = 512.0;
        camera.principalYPixel = 384.0;
        return camera;
    }

    std::vector<plabundle::Track> makeTracks(const std::vector<plabundle::FrameCamera>& cameras,
                                             const std::vector<Vec3>& points)
    {
        std::vector<plabundle::Track> tracks;
        for (const Vec3& point : points)
        {
            plabundle::Track track;
            track.initialPoint = point;
            for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
            {
                plabundle::Projection projection;
                if (!plabundle::projectWorldPoint(cameras[camera_index], point, &projection) ||
                    projection.pixel[0] < 16.0 || projection.pixel[0] > 1008.0 || projection.pixel[1] < 16.0 ||
                    projection.pixel[1] > 752.0)
                {
                    continue;
                }
                track.observations.push_back(
                    {static_cast<int>(camera_index), projection.pixel[0], projection.pixel[1], 1.0, 1.0});
            }
            if (track.observations.size() >= 3)
            {
                tracks.push_back(std::move(track));
            }
        }
        return tracks;
    }

    std::vector<Vec3> aerialPoints()
    {
        std::vector<Vec3> points;
        for (int row = -8; row <= 8; ++row)
        {
            for (int column = -10; column <= 10; ++column)
            {
                points.push_back({0.65 * column, 0.60 * row, 0.15 * std::sin(column * 0.35) * std::cos(row * 0.30)});
            }
        }
        return points;
    }

    std::vector<plabundle::FrameCamera> aerialCameras()
    {
        std::vector<plabundle::FrameCamera> cameras;
        for (int row = -2; row <= 2; ++row)
        {
            for (int column = -2; column <= 2; ++column)
            {
                const Vec3 center{{column * 2.2, row * 2.2, 15.0}};
                cameras.push_back(makeLookAtCamera(center, {center[0], center[1], 0.0}));
            }
        }
        return cameras;
    }

    std::vector<Vec3> orbitalPoints()
    {
        std::vector<Vec3> points;
        for (int depth = -2; depth <= 2; ++depth)
        {
            for (int row = -6; row <= 6; ++row)
            {
                for (int column = -6; column <= 6; ++column)
                {
                    points.push_back({0.78 * column, 0.68 * row, 0.70 * depth + 0.10 * std::sin(column + row)});
                }
            }
        }
        return points;
    }

    std::vector<plabundle::FrameCamera> orbitalCameras()
    {
        std::vector<plabundle::FrameCamera> cameras;
        constexpr int count = 24;
        constexpr double pi = 3.14159265358979323846;
        for (int index = 0; index < count; ++index)
        {
            const double angle = 2.0 * pi * index / count;
            const Vec3 center{{12.0 * std::cos(angle), 12.0 * std::sin(angle), 2.4 * std::sin(2.0 * angle)}};
            cameras.push_back(makeLookAtCamera(center, {0.0, 0.0, 0.0}, 620.0));
        }
        return cameras;
    }

    bool enabled(const plabundle::AdaptiveCameraModelAssessment& assessment, plabundle::IntrinsicParameter parameter)
    {
        return assessment.enabled[static_cast<std::size_t>(parameter)];
    }

} // namespace

TEST(PlaBundleAdaptiveCameraModelTest, UnanchoredParallelBlockKeepsIntrinsicsFixed)
{
    plabundle::Problem problem;
    problem.cameras = aerialCameras();
    problem.tracks = makeTracks(problem.cameras, aerialPoints());
    ASSERT_GT(problem.tracks.size(), 100U);

    const plabundle::AdaptiveCameraModelAssessment assessment = plabundle::assessAdaptiveCameraModel(problem);
    ASSERT_TRUE(assessment.valid);
    EXPECT_GT(assessment.opticalAxisConcentration, 0.99);
    EXPECT_FALSE(assessment.hasAbsoluteGeometryConstraint);
    EXPECT_TRUE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_EQ(plabundle::enabledIntrinsicParameterCount(assessment.enabled), 0);
    EXPECT_EQ(assessment.modelName, "fixed");
}

TEST(PlaBundleAdaptiveCameraModelTest, AbsoluteControlReleasesLowOrderAerialModel)
{
    plabundle::Problem problem;
    problem.cameras = aerialCameras();
    problem.tracks = makeTracks(problem.cameras, aerialPoints());
    ASSERT_GT(problem.tracks.size(), 100U);
    for (plabundle::Track& track : problem.tracks)
    {
        plabundle::ControlPointConstraint control;
        control.point = track.initialPoint;
        control.sigmaMeters = std::numeric_limits<double>::quiet_NaN();
        control.uncertainty = plabundle::ControlPointUncertainty::Covariance;
        control.uncertaintyMatrix = {4.0e-4, 1.0e-4, 0.0, 1.0e-4, 9.0e-4, 0.0, 0.0, 0.0, 1.6e-3};
        track.controlPointConstraints.push_back(control);
    }

    plabundle::Options options;
    const plabundle::AdaptiveCameraModelAssessment assessment = plabundle::assessAdaptiveCameraModel(problem, &options);
    ASSERT_TRUE(assessment.valid);
    EXPECT_TRUE(assessment.hasAbsoluteGeometryConstraint);
    EXPECT_FALSE(assessment.unanchoredParallelAerialGuardApplied);
    EXPECT_TRUE(enabled(assessment, plabundle::IntrinsicParameter::FocalLength));
    EXPECT_TRUE(enabled(assessment, plabundle::IntrinsicParameter::RadialK1));
}

TEST(PlaBundleAdaptiveCameraModelTest, ConvergentOrbitReleasesAnExtendedModel)
{
    plabundle::Problem problem;
    problem.cameras = orbitalCameras();
    problem.tracks = makeTracks(problem.cameras, orbitalPoints());
    ASSERT_GT(problem.tracks.size(), 200U);

    const plabundle::AdaptiveCameraModelAssessment assessment = plabundle::assessAdaptiveCameraModel(problem);
    ASSERT_TRUE(assessment.valid);
    EXPECT_LT(assessment.opticalAxisConcentration, 0.20);
    EXPECT_GT(assessment.geometryStrength, 0.75);
    EXPECT_TRUE(enabled(assessment, plabundle::IntrinsicParameter::FocalLength));
    EXPECT_TRUE(enabled(assessment, plabundle::IntrinsicParameter::RadialK1));
    EXPECT_TRUE(enabled(assessment, plabundle::IntrinsicParameter::FocalAspectRatio));
    EXPECT_TRUE(enabled(assessment, plabundle::IntrinsicParameter::RadialK2));
    EXPECT_GT(plabundle::enabledIntrinsicParameterCount(assessment.enabled), 2);
}

TEST(PlaBundleAdaptiveCameraModelTest, AppliesCallerMaskAndRestoresInactiveParameters)
{
    plabundle::AdaptiveCameraModelAssessment assessment;
    assessment.valid = true;
    assessment.enabled.fill(true);
    plabundle::SolveOptions options;
    options.calibration.refineSharedFocalLength = true;
    options.calibration.refineSharedFocalAspectRatio = true;
    options.calibration.refineSharedPrincipalPoint = true;
    options.calibration.refineSharedRadialDistortion = true;
    options.calibration.refineSharedHighOrderDistortion = true;
    options.calibration.useSharedIntrinsicParameterMask = true;
    options.calibration.sharedIntrinsicParameterMask.fill(false);
    options.calibration
        .sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::FocalLength)] = true;
    options.calibration
        .sharedIntrinsicParameterMask[static_cast<std::size_t>(plabundle::IntrinsicParameter::RadialK1)] = true;

    ASSERT_TRUE(plabundle::applyAdaptiveCameraModel(assessment, &options));
    EXPECT_EQ(plabundle::enabledIntrinsicParameterCount(options.calibration.sharedIntrinsicParameterMask), 2);

    std::vector<plabundle::FrameCamera> references = aerialCameras();
    std::vector<plabundle::FrameCamera> current = references;
    for (plabundle::FrameCamera& camera : current)
    {
        camera.focalXPixels = 840.0;
        camera.focalYPixels = 856.8;
        camera.principalXPixel = 524.0;
        camera.principalYPixel = 371.0;
        camera.distortion = {-0.08, -0.04, 0.01, 0.002, -0.003};
    }
    plabundle::IntrinsicParameterMask active{};
    active[static_cast<std::size_t>(plabundle::IntrinsicParameter::FocalLength)] = true;
    ASSERT_TRUE(plabundle::restoreInactiveAdaptiveIntrinsics(&current, references, active));
    for (std::size_t index = 0; index < current.size(); ++index)
    {
        EXPECT_DOUBLE_EQ(current[index].focalXPixels, 840.0);
        EXPECT_DOUBLE_EQ(current[index].focalYPixels, 840.0);
        EXPECT_DOUBLE_EQ(current[index].principalXPixel, references[index].principalXPixel);
        EXPECT_DOUBLE_EQ(current[index].principalYPixel, references[index].principalYPixel);
        EXPECT_EQ(current[index].distortion.k1, references[index].distortion.k1);
        EXPECT_EQ(current[index].distortion.k2, references[index].distortion.k2);
        EXPECT_EQ(current[index].distortion.k3, references[index].distortion.k3);
        EXPECT_EQ(current[index].distortion.p1, references[index].distortion.p1);
        EXPECT_EQ(current[index].distortion.p2, references[index].distortion.p2);
    }
}
