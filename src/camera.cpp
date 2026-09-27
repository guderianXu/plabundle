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

        struct NormalizedProjection
        {
            double x = 0.0;
            double y = 0.0;
            std::array<double, 4> byNormalized{};
            std::array<double, 16> byCoefficient{};
        };

        std::array<double, 8> projectionCoefficients(const FrameCamera& camera) noexcept
        {
            if (camera.projectionModel == FrameProjectionModel::BrownConrady ||
                camera.projectionModel == FrameProjectionModel::RollingShutter)
            {
                return {{camera.distortion.k1,
                         camera.distortion.k2,
                         camera.distortion.k3,
                         camera.distortion.p1,
                         camera.distortion.p2,
                         camera.distortion.k4,
                         camera.distortion.p3,
                         camera.distortion.p4}};
            }
            std::array<double, 8> coefficients{};
            std::copy(camera.modelCoefficients.begin(), camera.modelCoefficients.end(), coefficients.begin());
            return coefficients;
        }

        bool brownProjection(double x,
                             double y,
                             const std::array<double, 8>& coefficients,
                             BrownTangentialConvention convention,
                             NormalizedProjection* projection) noexcept
        {
            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double r8 = r4 * r4;
            const double radial =
                1.0 + coefficients[0] * r2 + coefficients[1] * r4 + coefficients[2] * r6 + coefficients[5] * r8;
            const double radial_slope =
                coefficients[0] + 2.0 * coefficients[1] * r2 + 3.0 * coefficients[2] * r4 + 4.0 * coefficients[5] * r6;
            const double radial_x = 2.0 * x * radial_slope;
            const double radial_y = 2.0 * y * radial_slope;
            if (convention == BrownTangentialConvention::OpenCv)
            {
                projection->x = x * radial + 2.0 * coefficients[3] * x * y + coefficients[4] * (r2 + 2.0 * x * x);
                projection->y = y * radial + coefficients[3] * (r2 + 2.0 * y * y) + 2.0 * coefficients[4] * x * y;
                projection->byNormalized = {
                    {radial + x * radial_x + 2.0 * coefficients[3] * y + 6.0 * coefficients[4] * x,
                     x * radial_y + 2.0 * coefficients[3] * x + 2.0 * coefficients[4] * y,
                     y * radial_x + 2.0 * coefficients[3] * x + 2.0 * coefficients[4] * y,
                     radial + y * radial_y + 6.0 * coefficients[3] * y + 2.0 * coefficients[4] * x}};
                projection->byCoefficient = {{x * r2,
                                              x * r4,
                                              x * r6,
                                              2.0 * x * y,
                                              r2 + 2.0 * x * x,
                                              x * r8,
                                              0.0,
                                              0.0,
                                              y * r2,
                                              y * r4,
                                              y * r6,
                                              r2 + 2.0 * y * y,
                                              2.0 * x * y,
                                              y * r8,
                                              0.0,
                                              0.0}};
                return true;
            }

            const double base_x = coefficients[3] * (3.0 * x * x + y * y) + 2.0 * coefficients[4] * x * y;
            const double base_y = 2.0 * coefficients[3] * x * y + coefficients[4] * (x * x + 3.0 * y * y);
            const double tangential_scale = 1.0 + coefficients[6] * r2 + coefficients[7] * r4;
            const double tangential_slope = coefficients[6] + 2.0 * coefficients[7] * r2;
            const double scale_x = 2.0 * x * tangential_slope;
            const double scale_y = 2.0 * y * tangential_slope;
            const double base_x_by_x = 6.0 * coefficients[3] * x + 2.0 * coefficients[4] * y;
            const double base_x_by_y = 2.0 * coefficients[3] * y + 2.0 * coefficients[4] * x;
            const double base_y_by_x = 2.0 * coefficients[3] * y + 2.0 * coefficients[4] * x;
            const double base_y_by_y = 2.0 * coefficients[3] * x + 6.0 * coefficients[4] * y;
            projection->x = x * radial + base_x * tangential_scale;
            projection->y = y * radial + base_y * tangential_scale;
            projection->byNormalized = {{radial + x * radial_x + base_x_by_x * tangential_scale + base_x * scale_x,
                                         x * radial_y + base_x_by_y * tangential_scale + base_x * scale_y,
                                         y * radial_x + base_y_by_x * tangential_scale + base_y * scale_x,
                                         radial + y * radial_y + base_y_by_y * tangential_scale + base_y * scale_y}};
            projection->byCoefficient = {{x * r2,
                                          x * r4,
                                          x * r6,
                                          (3.0 * x * x + y * y) * tangential_scale,
                                          2.0 * x * y * tangential_scale,
                                          x * r8,
                                          base_x * r2,
                                          base_x * r4,
                                          y * r2,
                                          y * r4,
                                          y * r6,
                                          2.0 * x * y * tangential_scale,
                                          (x * x + 3.0 * y * y) * tangential_scale,
                                          y * r8,
                                          base_y * r2,
                                          base_y * r4}};
            return true;
        }

        bool radialProjection(FrameProjectionModel model,
                              double x,
                              double y,
                              const std::array<double, 8>& coefficients,
                              NormalizedProjection* projection) noexcept
        {
            const double radius = std::hypot(x, y);
            if (radius < 1.0e-12)
            {
                projection->x = x;
                projection->y = y;
                projection->byNormalized = {{1.0, 0.0, 0.0, 1.0}};
                return true;
            }
            const double theta = std::atan(radius);
            const double theta2 = theta * theta;
            std::array<double, 5> theta_powers{};
            double power = theta2;
            double polynomial = 1.0;
            double polynomial_derivative = 0.0;
            for (std::size_t index = 0; index < 5; ++index)
            {
                theta_powers[index] = power;
                polynomial += coefficients[index] * power;
                polynomial_derivative += coefficients[index] * static_cast<double>(2 * (index + 1)) * power / theta;
                power *= theta2;
            }
            double base = theta;
            double base_derivative = 1.0;
            if (model == FrameProjectionModel::Fisheye)
            {
                base = 2.0 * std::tan(0.5 * theta);
                const double cosine = std::cos(0.5 * theta);
                base_derivative = 1.0 / (cosine * cosine);
            }
            else if (model == FrameProjectionModel::Equisolid)
            {
                base = 2.0 * std::sin(0.5 * theta);
                base_derivative = std::cos(0.5 * theta);
            }
            const double rho = base * polynomial;
            const double rho_by_theta = base_derivative * polynomial + base * polynomial_derivative;
            const double rho_by_radius = rho_by_theta / (1.0 + radius * radius);
            const double scale = rho / radius;
            const double scale_by_radius = (rho_by_radius * radius - rho) / (radius * radius);
            const double scale_by_x = scale_by_radius * x / radius;
            const double scale_by_y = scale_by_radius * y / radius;
            projection->x = x * scale;
            projection->y = y * scale;
            projection->byNormalized = {
                {scale + x * scale_by_x, x * scale_by_y, y * scale_by_x, scale + y * scale_by_y}};
            for (std::size_t index = 0; index < 5; ++index)
            {
                const double rho_by_coefficient = base * theta_powers[index];
                projection->byCoefficient[index] = x * rho_by_coefficient / radius;
                projection->byCoefficient[index + 8] = y * rho_by_coefficient / radius;
            }
            return true;
        }

        bool
        angularProjection(FrameProjectionModel model, double x, double y, NormalizedProjection* projection) noexcept
        {
            const double horizontal = std::sqrt(1.0 + x * x);
            projection->x = std::atan(x);
            projection->byNormalized[0] = 1.0 / (1.0 + x * x);
            if (model == FrameProjectionModel::Spherical)
            {
                const double denominator = 1.0 + x * x + y * y;
                projection->y = std::atan2(y, horizontal);
                projection->byNormalized[2] = -x * y / (horizontal * denominator);
                projection->byNormalized[3] = horizontal / denominator;
            }
            else
            {
                projection->y = y / horizontal;
                projection->byNormalized[2] = -x * y / (horizontal * horizontal * horizontal);
                projection->byNormalized[3] = 1.0 / horizontal;
            }
            return true;
        }

        bool
        normalizedProjection(const FrameCamera& camera, double x, double y, NormalizedProjection* projection) noexcept
        {
            *projection = NormalizedProjection{};
            const auto coefficients = projectionCoefficients(camera);
            switch (camera.projectionModel)
            {
            case FrameProjectionModel::BrownConrady:
            case FrameProjectionModel::RollingShutter:
                return brownProjection(x, y, coefficients, camera.brownTangentialConvention, projection);
            case FrameProjectionModel::Fisheye:
            case FrameProjectionModel::Equidistant:
            case FrameProjectionModel::Equisolid:
                return radialProjection(camera.projectionModel, x, y, coefficients, projection);
            case FrameProjectionModel::Spherical:
            case FrameProjectionModel::Cylindrical:
                return angularProjection(camera.projectionModel, x, y, projection);
            }
            return false;
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
            !finite(distortion.p2) || !finite(distortion.k4) || !finite(distortion.p3) || !finite(distortion.p4) ||
            !finite(camera.skewPixels))
        {
            setError(error, "camera distortion coefficients must be finite");
            return false;
        }
        switch (camera.brownTangentialConvention)
        {
        case BrownTangentialConvention::OpenCv:
            if (distortion.p3 != 0.0 || distortion.p4 != 0.0)
            {
                setError(error, "p3/p4 require the Metashape tangential convention");
                return false;
            }
            break;
        case BrownTangentialConvention::Metashape:
            break;
        default:
            setError(error, "camera tangential-distortion convention is not supported");
            return false;
        }
        switch (camera.projectionModel)
        {
        case FrameProjectionModel::BrownConrady:
        case FrameProjectionModel::Fisheye:
        case FrameProjectionModel::Equidistant:
        case FrameProjectionModel::Equisolid:
        case FrameProjectionModel::Spherical:
        case FrameProjectionModel::Cylindrical:
        case FrameProjectionModel::RollingShutter:
            break;
        default:
            setError(error, "camera projection model is not supported");
            return false;
        }
        if (!finiteArray(camera.modelCoefficients))
        {
            setError(error, "camera model coefficients must be finite");
            return false;
        }
        if (!finite(camera.rollingShutter.referenceLinePixels) || !finite(camera.rollingShutter.secondsPerLine) ||
            !finiteArray(camera.rollingShutter.linearVelocityWorldMetersPerSecond) ||
            !finiteArray(camera.rollingShutter.angularVelocityCameraRadiansPerSecond) ||
            (camera.projectionModel == FrameProjectionModel::RollingShutter &&
             camera.rollingShutter.secondsPerLine == 0.0))
        {
            setError(error, "rolling-shutter timing and motion must be finite with a non-zero line period");
            return false;
        }
        if (camera.imageSize && !camera.imageSize->valid())
        {
            setError(error, "camera image size must be positive when present");
            return false;
        }
        return true;
    }

    CameraParameterBlock cameraParameterBlock(const FrameCamera& camera) noexcept
    {
        CameraParameterBlock parameters{};
        parameters[0] = camera.focalXPixels;
        parameters[1] = camera.focalXPixels > 0.0 && camera.focalYPixels > 0.0
                            ? std::log(camera.focalYPixels / camera.focalXPixels)
                            : std::numeric_limits<double>::quiet_NaN();
        parameters[2] = camera.principalXPixel;
        parameters[3] = camera.principalYPixel;
        const auto coefficients = projectionCoefficients(camera);
        std::copy(coefficients.begin(), coefficients.begin() + 5, parameters.begin() + 4);
        if (camera.projectionModel == FrameProjectionModel::BrownConrady ||
            camera.projectionModel == FrameProjectionModel::RollingShutter)
        {
            parameters[9] = camera.skewPixels;
            parameters[10] = coefficients[5];
            parameters[11] = coefficients[6];
            parameters[12] = coefficients[7];
        }
        return parameters;
    }

    CameraParameterMask supportedCameraParameters(FrameProjectionModel model) noexcept
    {
        CameraParameterMask mask{};
        mask[0] = mask[1] = mask[2] = mask[3] = true;
        if (model == FrameProjectionModel::BrownConrady || model == FrameProjectionModel::RollingShutter ||
            model == FrameProjectionModel::Fisheye || model == FrameProjectionModel::Equidistant ||
            model == FrameProjectionModel::Equisolid)
        {
            std::fill(mask.begin() + 4, mask.begin() + 9, true);
        }
        if (model == FrameProjectionModel::BrownConrady || model == FrameProjectionModel::RollingShutter)
        {
            mask[9] = true;
            mask[10] = true;
        }
        return mask;
    }

    CameraParameterMask supportedCameraParameters(const FrameCamera& camera) noexcept
    {
        CameraParameterMask mask = supportedCameraParameters(camera.projectionModel);
        if ((camera.projectionModel == FrameProjectionModel::BrownConrady ||
             camera.projectionModel == FrameProjectionModel::RollingShutter) &&
            camera.brownTangentialConvention == BrownTangentialConvention::Metashape)
        {
            mask[11] = true;
            mask[12] = true;
        }
        return mask;
    }

    bool applyCameraParameterBlock(FrameCamera* camera,
                                   const CameraParameterBlock& parameters,
                                   const CameraParameterMask& mask,
                                   std::string* error) noexcept
    {
        if (!camera || !validateFrameCamera(*camera, error) || !finiteArray(parameters))
        {
            if (!camera)
            {
                setError(error, "camera output is null");
            }
            else if (!finiteArray(parameters))
            {
                setError(error, "camera parameter block must contain finite values");
            }
            return false;
        }
        const CameraParameterMask supported = supportedCameraParameters(*camera);
        for (std::size_t index = 0; index < mask.size(); ++index)
        {
            if (mask[index] && !supported[index])
            {
                setError(error, "camera parameter mask enables a coefficient unsupported by the projection model");
                return false;
            }
        }

        FrameCamera candidate = *camera;
        CameraParameterBlock effective = cameraParameterBlock(candidate);
        for (std::size_t index = 0; index < mask.size(); ++index)
        {
            if (mask[index])
            {
                effective[index] = parameters[index];
            }
        }
        candidate.focalXPixels = effective[0];
        candidate.focalYPixels = effective[0] * std::exp(effective[1]);
        candidate.principalXPixel = effective[2];
        candidate.principalYPixel = effective[3];
        if (candidate.projectionModel == FrameProjectionModel::BrownConrady ||
            candidate.projectionModel == FrameProjectionModel::RollingShutter)
        {
            candidate.distortion.k1 = effective[4];
            candidate.distortion.k2 = effective[5];
            candidate.distortion.k3 = effective[6];
            candidate.distortion.p1 = effective[7];
            candidate.distortion.p2 = effective[8];
            candidate.skewPixels = effective[9];
            candidate.distortion.k4 = effective[10];
            if (candidate.brownTangentialConvention == BrownTangentialConvention::Metashape)
            {
                candidate.distortion.p3 = effective[11];
                candidate.distortion.p4 = effective[12];
            }
        }
        else
        {
            std::copy(effective.begin() + 4, effective.end(), candidate.modelCoefficients.begin());
        }
        if (!validateFrameCamera(candidate, error))
        {
            return false;
        }
        *camera = candidate;
        return true;
    }

    bool applyMetashapeFrameCalibration(FrameCamera* camera,
                                        const MetashapeFrameCalibration& calibration,
                                        std::string* error) noexcept
    {
        if (!camera)
        {
            setError(error, "camera output is null");
            return false;
        }
        if (camera->projectionModel != FrameProjectionModel::BrownConrady &&
            camera->projectionModel != FrameProjectionModel::RollingShutter)
        {
            setError(error, "Metashape calibration requires a Brown or rolling-shutter frame model");
            return false;
        }
        const std::array<double, 13> values{{calibration.f,
                                             calibration.cx,
                                             calibration.cy,
                                             calibration.b1,
                                             calibration.b2,
                                             calibration.k1,
                                             calibration.k2,
                                             calibration.k3,
                                             calibration.k4,
                                             calibration.p1,
                                             calibration.p2,
                                             calibration.p3,
                                             calibration.p4}};
        if (!finiteArray(values) || calibration.f <= 0.0 || calibration.f + calibration.b1 <= 0.0)
        {
            setError(error, "Metashape calibration must be finite with positive f and f+b1");
            return false;
        }
        FrameCamera candidate = *camera;
        candidate.focalXPixels = calibration.f + calibration.b1;
        candidate.focalYPixels = calibration.f;
        candidate.principalXPixel = calibration.cx;
        candidate.principalYPixel = calibration.cy;
        candidate.skewPixels = calibration.b2;
        candidate.distortion = {calibration.k1,
                                calibration.k2,
                                calibration.k3,
                                calibration.p1,
                                calibration.p2,
                                calibration.k4,
                                calibration.p3,
                                calibration.p4};
        candidate.brownTangentialConvention = BrownTangentialConvention::Metashape;
        if (!validateFrameCamera(candidate, error))
        {
            return false;
        }
        *camera = candidate;
        return true;
    }

    bool metashapeFrameCalibration(const FrameCamera& camera,
                                   MetashapeFrameCalibration* calibration,
                                   std::string* error) noexcept
    {
        if (!calibration)
        {
            setError(error, "Metashape calibration output is null");
            return false;
        }
        if (!validateFrameCamera(camera, error) ||
            (camera.projectionModel != FrameProjectionModel::BrownConrady &&
             camera.projectionModel != FrameProjectionModel::RollingShutter) ||
            camera.brownTangentialConvention != BrownTangentialConvention::Metashape)
        {
            if (validateFrameCamera(camera) && camera.brownTangentialConvention != BrownTangentialConvention::Metashape)
            {
                setError(error, "camera does not use the Metashape tangential convention");
            }
            return false;
        }
        *calibration = {camera.focalYPixels,
                        camera.principalXPixel,
                        camera.principalYPixel,
                        camera.focalXPixels - camera.focalYPixels,
                        camera.skewPixels,
                        camera.distortion.k1,
                        camera.distortion.k2,
                        camera.distortion.k3,
                        camera.distortion.k4,
                        camera.distortion.p1,
                        camera.distortion.p2,
                        camera.distortion.p3,
                        camera.distortion.p4};
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

    std::array<double, 3>
    worldToCameraPointAtLine(const FrameCamera& camera, const std::array<double, 3>& world, double linePixels) noexcept
    {
        FrameCamera effective;
        if (!cameraAtLine(camera, linePixels, &effective))
        {
            const double invalid = std::numeric_limits<double>::quiet_NaN();
            return {{invalid, invalid, invalid}};
        }
        return worldToCameraPoint(effective, world);
    }

    bool cameraAtLine(const FrameCamera& camera, double linePixels, FrameCamera* effectiveCamera) noexcept
    {
        if (!effectiveCamera || !validateFrameCamera(camera) || !std::isfinite(linePixels))
        {
            return false;
        }
        *effectiveCamera = camera;
        if (camera.projectionModel != FrameProjectionModel::RollingShutter)
        {
            return true;
        }
        const double delta_time =
            (linePixels - camera.rollingShutter.referenceLinePixels) * camera.rollingShutter.secondsPerLine;
        std::array<double, 6> delta{};
        for (int axis = 0; axis < 3; ++axis)
        {
            delta[axis] = camera.rollingShutter.angularVelocityCameraRadiansPerSecond[axis] * delta_time;
            delta[axis + 3] = camera.rollingShutter.linearVelocityWorldMetersPerSecond[axis] * delta_time;
        }
        if (!std::isfinite(delta_time) || !applyPoseDelta(effectiveCamera, delta))
        {
            return false;
        }
        return true;
    }

    double positiveDepth(const FrameCamera& camera, const std::array<double, 3>& world) noexcept
    {
        const std::array<double, 3> camera_point = worldToCameraPoint(camera, world);
        return camera.depthAxisFlipped ? -camera_point[2] : camera_point[2];
    }

    bool
    projectWorldPoint(const FrameCamera& camera, const std::array<double, 3>& world, Projection* projection) noexcept
    {
        return projectWorldPointAtLine(camera, world, camera.rollingShutter.referenceLinePixels, projection);
    }

    bool projectWorldPointAtLine(const FrameCamera& camera,
                                 const std::array<double, 3>& world,
                                 double linePixels,
                                 Projection* projection) noexcept
    {
        if (!projection || !validateFrameCamera(camera) || !finiteArray(world) || !finite(linePixels))
        {
            return false;
        }
        ProjectionLinearization linearization;
        if (!linearizeCameraPoint(camera, worldToCameraPointAtLine(camera, world, linePixels), &linearization))
        {
            return false;
        }
        *projection = linearization.projection;
        return true;
    }

    bool linearizeCameraPoint(const FrameCamera& camera,
                              const std::array<double, 3>& cameraPoint,
                              ProjectionLinearization* linearization) noexcept
    {
        if (!linearization || !validateFrameCamera(camera) || !finiteArray(cameraPoint))
        {
            return false;
        }
        const double depth = camera.depthAxisFlipped ? -cameraPoint[2] : cameraPoint[2];
        if (!finite(depth) || depth <= kMinimumDepth || std::abs(cameraPoint[2]) <= kMinimumDepth)
        {
            return false;
        }

        const double x = cameraPoint[0] / cameraPoint[2];
        const double y = cameraPoint[1] / cameraPoint[2];
        NormalizedProjection normalized;
        if (!normalizedProjection(camera, x, y, &normalized))
        {
            return false;
        }
        const double u_sign = static_cast<double>(camera.uAxisSign);
        const double v_sign = static_cast<double>(camera.vAxisSign);
        const double u_scale = u_sign * camera.focalXPixels;
        const double skew_scale = u_sign * camera.skewPixels;
        const double v_scale = static_cast<double>(camera.vAxisSign) * camera.focalYPixels;
        ProjectionLinearization output;
        output.projection.pixel[0] = u_scale * normalized.x + skew_scale * normalized.y + camera.principalXPixel;
        output.projection.pixel[1] = v_scale * normalized.y + camera.principalYPixel;
        output.projection.positiveDepth = depth;

        const double inverse_depth = 1.0 / cameraPoint[2];
        const std::array<double, 6> normalized_by_camera{
            {inverse_depth, 0.0, -x * inverse_depth, 0.0, inverse_depth, -y * inverse_depth}};
        for (int output_axis = 0; output_axis < 2; ++output_axis)
        {
            const double normalized_x_derivative =
                output_axis == 0 ? u_scale * normalized.byNormalized[0] + skew_scale * normalized.byNormalized[2]
                                 : v_scale * normalized.byNormalized[2];
            const double normalized_y_derivative =
                output_axis == 0 ? u_scale * normalized.byNormalized[1] + skew_scale * normalized.byNormalized[3]
                                 : v_scale * normalized.byNormalized[3];
            for (int camera_axis = 0; camera_axis < 3; ++camera_axis)
            {
                output.cameraPointJacobian[output_axis * 3 + camera_axis] =
                    normalized_x_derivative * normalized_by_camera[camera_axis] +
                    normalized_y_derivative * normalized_by_camera[3 + camera_axis];
            }
        }

        const double aspect = camera.focalYPixels / camera.focalXPixels;
        output.parameterJacobian[0] = u_sign * normalized.x;
        output.parameterJacobian[kCameraParameterCount] = v_sign * aspect * normalized.y;
        output.parameterJacobian[kCameraParameterCount + 1] = v_scale * normalized.y;
        output.parameterJacobian[2] = 1.0;
        output.parameterJacobian[kCameraParameterCount + 3] = 1.0;
        output.parameterJacobian[9] = u_sign * normalized.y;
        const CameraParameterMask supported = supportedCameraParameters(camera);
        constexpr std::array<std::size_t, 8> coefficient_parameters{{4, 5, 6, 7, 8, 10, 11, 12}};
        for (std::size_t coefficient = 0; coefficient < coefficient_parameters.size(); ++coefficient)
        {
            const std::size_t parameter = coefficient_parameters[coefficient];
            if (supported[parameter])
            {
                const double normalized_x = normalized.byCoefficient[coefficient];
                const double normalized_y = normalized.byCoefficient[coefficient + 8];
                output.parameterJacobian[parameter] = u_scale * normalized_x + skew_scale * normalized_y;
                output.parameterJacobian[kCameraParameterCount + parameter] = v_scale * normalized_y;
            }
        }

        if (!finiteArray(output.projection.pixel) || !finiteArray(output.cameraPointJacobian) ||
            !finiteArray(output.parameterJacobian))
        {
            return false;
        }
        *linearization = output;
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
