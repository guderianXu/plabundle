#include <plabundle/camera.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace plabundle
{
    namespace
    {
        constexpr double kMinimumDepth = 1.0e-9;
        constexpr double kRotationTolerance = 1.0e-8;

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        template <std::size_t Size> bool finiteArray(const std::array<double, Size>& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return finite(value); });
        }

        void setError(std::string* error, const char* message) noexcept
        {
            if (!error)
            {
                return;
            }
            try
            {
                *error = message;
            }
            catch (...)
            {
            }
        }

        bool validateRotation(const std::array<double, 9>& rotation) noexcept
        {
            if (!finiteArray(rotation))
            {
                return false;
            }
            const auto dotRow = [&rotation](int first, int second)
            {
                return rotation[static_cast<std::size_t>(first)] * rotation[static_cast<std::size_t>(second)] +
                       rotation[static_cast<std::size_t>(first + 1)] * rotation[static_cast<std::size_t>(second + 1)] +
                       rotation[static_cast<std::size_t>(first + 2)] * rotation[static_cast<std::size_t>(second + 2)];
            };
            const double determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                                       rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                                       rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
            return std::abs(dotRow(0, 0) - 1.0) <= kRotationTolerance &&
                   std::abs(dotRow(3, 3) - 1.0) <= kRotationTolerance &&
                   std::abs(dotRow(6, 6) - 1.0) <= kRotationTolerance && std::abs(dotRow(0, 3)) <= kRotationTolerance &&
                   std::abs(dotRow(0, 6)) <= kRotationTolerance && std::abs(dotRow(3, 6)) <= kRotationTolerance &&
                   std::abs(determinant - 1.0) <= kRotationTolerance;
        }

        std::array<double, 9> rotationFromDelta(const std::array<double, 6>& delta) noexcept
        {
            const double wx = delta[0];
            const double wy = delta[1];
            const double wz = delta[2];
            const double theta_squared = wx * wx + wy * wy + wz * wz;
            if (theta_squared < 1.0e-20)
            {
                return {1.0, -wz, wy, wz, 1.0, -wx, -wy, wx, 1.0};
            }
            const double theta = std::sqrt(theta_squared);
            const double sine_over_theta = std::sin(theta) / theta;
            const double one_minus_cosine_over_theta_squared = (1.0 - std::cos(theta)) / theta_squared;
            return {1.0 - one_minus_cosine_over_theta_squared * (wy * wy + wz * wz),
                    one_minus_cosine_over_theta_squared * wx * wy - sine_over_theta * wz,
                    one_minus_cosine_over_theta_squared * wx * wz + sine_over_theta * wy,
                    one_minus_cosine_over_theta_squared * wx * wy + sine_over_theta * wz,
                    1.0 - one_minus_cosine_over_theta_squared * (wx * wx + wz * wz),
                    one_minus_cosine_over_theta_squared * wy * wz - sine_over_theta * wx,
                    one_minus_cosine_over_theta_squared * wx * wz - sine_over_theta * wy,
                    one_minus_cosine_over_theta_squared * wy * wz + sine_over_theta * wx,
                    1.0 - one_minus_cosine_over_theta_squared * (wx * wx + wy * wy)};
        }

        std::array<double, 9> multiply(const std::array<double, 9>& left, const std::array<double, 9>& right) noexcept
        {
            std::array<double, 9> result{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int inner = 0; inner < 3; ++inner)
                    {
                        result[static_cast<std::size_t>(row * 3 + column)] +=
                            left[static_cast<std::size_t>(row * 3 + inner)] *
                            right[static_cast<std::size_t>(inner * 3 + column)];
                    }
                }
            }
            return result;
        }

    } // namespace

    bool validateFrameCamera(const FrameCamera& camera, std::string* error) noexcept
    {
        if (error)
        {
            try
            {
                error->clear();
            }
            catch (...)
            {
            }
        }
        if (!validateRotation(camera.cameraToWorldRotation))
        {
            setError(error, "camera rotation must be a finite proper orthonormal matrix");
            return false;
        }
        if (!finiteArray(camera.cameraCenter))
        {
            setError(error, "camera center must contain finite values");
            return false;
        }
        if (!finite(camera.focalXPixels) || !finite(camera.focalYPixels) || camera.focalXPixels <= 0.0 ||
            camera.focalYPixels <= 0.0 || !finite(camera.principalXPixel) || !finite(camera.principalYPixel) ||
            !finite(camera.pixelPitchMillimeters) || camera.pixelPitchMillimeters <= 0.0)
        {
            setError(error, "camera intrinsics must be finite with positive focal lengths and pixel pitch");
            return false;
        }
        if (camera.uAxisSign != 1 && camera.uAxisSign != -1)
        {
            setError(error, "camera u-axis sign must be +1 or -1");
            return false;
        }
        if (camera.vAxisSign != 1 && camera.vAxisSign != -1)
        {
            setError(error, "camera v-axis sign must be +1 or -1");
            return false;
        }
        const BrownConradyDistortion& distortion = camera.distortion;
        if (!finite(distortion.k1) || !finite(distortion.k2) || !finite(distortion.k3) || !finite(distortion.p1) ||
            !finite(distortion.p2))
        {
            setError(error, "camera distortion coefficients must be finite");
            return false;
        }
        if (camera.imageSize && !camera.imageSize->valid())
        {
            setError(error, "camera image size must be positive when present");
            return false;
        }
        return true;
    }

    std::array<double, 3> worldToCameraPoint(const FrameCamera& camera, const std::array<double, 3>& world) noexcept
    {
        const double dx = world[0] - camera.cameraCenter[0];
        const double dy = world[1] - camera.cameraCenter[1];
        const double dz = world[2] - camera.cameraCenter[2];
        return {camera.cameraToWorldRotation[0] * dx + camera.cameraToWorldRotation[3] * dy +
                    camera.cameraToWorldRotation[6] * dz,
                camera.cameraToWorldRotation[1] * dx + camera.cameraToWorldRotation[4] * dy +
                    camera.cameraToWorldRotation[7] * dz,
                camera.cameraToWorldRotation[2] * dx + camera.cameraToWorldRotation[5] * dy +
                    camera.cameraToWorldRotation[8] * dz};
    }

    double positiveDepth(const FrameCamera& camera, const std::array<double, 3>& world) noexcept
    {
        const std::array<double, 3> camera_point = worldToCameraPoint(camera, world);
        return camera.depthAxisFlipped ? -camera_point[2] : camera_point[2];
    }

    bool
    projectWorldPoint(const FrameCamera& camera, const std::array<double, 3>& world, Projection* projection) noexcept
    {
        if (!projection || !validateFrameCamera(camera) || !finiteArray(world))
        {
            return false;
        }
        const std::array<double, 3> camera_point = worldToCameraPoint(camera, world);
        const double depth = camera.depthAxisFlipped ? -camera_point[2] : camera_point[2];
        if (!finite(depth) || depth <= kMinimumDepth || !finite(camera_point[2]) ||
            std::abs(camera_point[2]) <= kMinimumDepth)
        {
            return false;
        }

        const double x = camera_point[0] / camera_point[2];
        const double y = camera_point[1] / camera_point[2];
        const double radius_squared = x * x + y * y;
        const BrownConradyDistortion& distortion = camera.distortion;
        const double radial = 1.0 + distortion.k1 * radius_squared + distortion.k2 * radius_squared * radius_squared +
                              distortion.k3 * radius_squared * radius_squared * radius_squared;
        const double distorted_x =
            x * radial + 2.0 * distortion.p1 * x * y + distortion.p2 * (radius_squared + 2.0 * x * x);
        const double distorted_y =
            y * radial + distortion.p1 * (radius_squared + 2.0 * y * y) + 2.0 * distortion.p2 * x * y;
        Projection result;
        result.pixel[0] =
            static_cast<double>(camera.uAxisSign) * camera.focalXPixels * distorted_x + camera.principalXPixel;
        result.pixel[1] =
            static_cast<double>(camera.vAxisSign) * camera.focalYPixels * distorted_y + camera.principalYPixel;
        result.positiveDepth = depth;
        if (!finite(result.pixel[0]) || !finite(result.pixel[1]))
        {
            return false;
        }
        *projection = result;
        return true;
    }

    bool applyPoseDelta(FrameCamera* camera, const std::array<double, 6>& delta, std::string* error) noexcept
    {
        if (!camera)
        {
            setError(error, "camera output is null");
            return false;
        }
        if (!validateFrameCamera(*camera, error) || !finiteArray(delta))
        {
            if (finiteArray(delta))
            {
                return false;
            }
            setError(error, "pose delta must contain finite values");
            return false;
        }

        FrameCamera candidate = *camera;
        candidate.cameraToWorldRotation = multiply(rotationFromDelta(delta), camera->cameraToWorldRotation);
        candidate.cameraCenter[0] += delta[3];
        candidate.cameraCenter[1] += delta[4];
        candidate.cameraCenter[2] += delta[5];
        if (!validateFrameCamera(candidate, error))
        {
            return false;
        }
        *camera = candidate;
        return true;
    }

} // namespace plabundle
