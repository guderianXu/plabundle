#include "linescan_internal.h"

#include <cmath>
#include <limits>

namespace plabundle::linescan
{
    namespace
    {

        using Matrix3 = std::array<double, 9>;
        using Vector3 = std::array<double, 3>;

        double vectorNorm(const Vector3& value)
        {
            return std::hypot(std::hypot(value[0], value[1]), value[2]);
        }

        bool finiteMatrix(const Matrix3& matrix)
        {
            for (double value : matrix)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
            }
            return true;
        }

        Vector3 multiply(const Matrix3& matrix, const Vector3& vector)
        {
            return {{matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
                     matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
                     matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]}};
        }

        Matrix3 multiply(const Matrix3& left, const Matrix3& right)
        {
            Matrix3 output{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int inner = 0; inner < 3; ++inner)
                    {
                        output[row * 3 + column] += left[row * 3 + inner] * right[inner * 3 + column];
                    }
                }
            }
            return output;
        }

        struct CorrectionRotation
        {
            Matrix3 value{};
            std::array<Matrix3, 3> derivatives{};
        };

        CorrectionRotation makeCorrectionRotation(const std::array<double, 6>& correction)
        {
            const double roll = correction[3];
            const double pitch = correction[4];
            const double yaw = correction[5];
            const double sr = std::sin(roll);
            const double cr = std::cos(roll);
            const double sp = std::sin(pitch);
            const double cp = std::cos(pitch);
            const double sy = std::sin(yaw);
            const double cy = std::cos(yaw);
            const Matrix3 rx{{1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr}};
            const Matrix3 ry{{cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp}};
            const Matrix3 rz{{cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0}};
            const Matrix3 drx{{0.0, 0.0, 0.0, 0.0, -sr, -cr, 0.0, cr, -sr}};
            const Matrix3 dry{{-sp, 0.0, cp, 0.0, 0.0, 0.0, -cp, 0.0, -sp}};
            const Matrix3 drz{{-sy, -cy, 0.0, cy, -sy, 0.0, 0.0, 0.0, 0.0}};

            CorrectionRotation rotation;
            rotation.value = multiply(multiply(rz, ry), rx);
            rotation.derivatives[0] = multiply(multiply(rz, ry), drx);
            rotation.derivatives[1] = multiply(multiply(rz, dry), rx);
            rotation.derivatives[2] = multiply(multiply(drz, ry), rx);
            return rotation;
        }

        bool validPose(const SensorPose& pose)
        {
            if (!finiteMatrix(pose.worldToSensorRotation))
            {
                return false;
            }
            for (double value : pose.centerMeters)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
            }
            for (int row = 0; row < 3; ++row)
            {
                double norm = 0.0;
                for (int column = 0; column < 3; ++column)
                {
                    norm += pose.worldToSensorRotation[row * 3 + column] * pose.worldToSensorRotation[row * 3 + column];
                }
                if (std::abs(norm - 1.0) > 1.0e-6)
                {
                    return false;
                }
            }
            for (int first = 0; first < 3; ++first)
            {
                for (int second = first + 1; second < 3; ++second)
                {
                    double dot = 0.0;
                    for (int column = 0; column < 3; ++column)
                    {
                        dot += pose.worldToSensorRotation[first * 3 + column] *
                               pose.worldToSensorRotation[second * 3 + column];
                    }
                    if (std::abs(dot) > 1.0e-6)
                    {
                        return false;
                    }
                }
            }
            const double determinant =
                pose.worldToSensorRotation[0] * (pose.worldToSensorRotation[4] * pose.worldToSensorRotation[8] -
                                                 pose.worldToSensorRotation[5] * pose.worldToSensorRotation[7]) -
                pose.worldToSensorRotation[1] * (pose.worldToSensorRotation[3] * pose.worldToSensorRotation[8] -
                                                 pose.worldToSensorRotation[5] * pose.worldToSensorRotation[6]) +
                pose.worldToSensorRotation[2] * (pose.worldToSensorRotation[3] * pose.worldToSensorRotation[7] -
                                                 pose.worldToSensorRotation[4] * pose.worldToSensorRotation[6]);
            return determinant > 0.0 && std::abs(determinant - 1.0) <= 1.0e-6;
        }

        bool validModelScalars(const CameraModel& model)
        {
            return std::isfinite(model.timing.referenceLinePixels) &&
                   std::isfinite(model.timing.referenceTimeSeconds) && std::isfinite(model.timing.secondsPerLine) &&
                   model.timing.secondsPerLine != 0.0 && std::isfinite(model.focalSamplePixels) &&
                   model.focalSamplePixels > 0.0 && std::isfinite(model.focalLinePixels) &&
                   model.focalLinePixels > 0.0 && std::isfinite(model.principalSamplePixels) &&
                   std::isfinite(model.detectorLinePixels) && std::isfinite(model.minimumDepthMeters) &&
                   model.minimumDepthMeters > 0.0;
        }

    } // namespace

    double LineTimingModel::timeAtLine(double linePixels) const noexcept
    {
        return referenceTimeSeconds + (linePixels - referenceLinePixels) * secondsPerLine;
    }

    bool prepareProjectionState(const CameraModel& model, double linePixels, ProjectionState* state)
    {
        if (!state || !model.trajectory || !validModelScalars(model) || !std::isfinite(linePixels))
        {
            return false;
        }
        ProjectionState candidate;
        candidate.acquisitionTimeSeconds = model.timing.timeAtLine(linePixels);
        if (!std::isfinite(candidate.acquisitionTimeSeconds) ||
            !model.trajectory->evaluate(candidate.acquisitionTimeSeconds, &candidate.nominalPose) ||
            !validPose(candidate.nominalPose))
        {
            return false;
        }
        *state = candidate;
        return true;
    }

    bool evaluateProjection(const CameraModel& model,
                            const ProjectionState& state,
                            const ImageObservation& observation,
                            const std::array<double, 6>& cameraCorrection,
                            const std::array<double, 3>& pointMeters,
                            ProjectionEvaluation* evaluation)
    {
        if (!evaluation || !validModelScalars(model) || !validPose(state.nominalPose) ||
            !std::isfinite(state.acquisitionTimeSeconds))
        {
            return false;
        }
        for (double value : cameraCorrection)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }
        for (double value : pointMeters)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }

        const Vector3 world_delta{{pointMeters[0] - state.nominalPose.centerMeters[0] - cameraCorrection[0],
                                   pointMeters[1] - state.nominalPose.centerMeters[1] - cameraCorrection[1],
                                   pointMeters[2] - state.nominalPose.centerMeters[2] - cameraCorrection[2]}};
        const Vector3 nominal_sensor = multiply(state.nominalPose.worldToSensorRotation, world_delta);
        const CorrectionRotation correction_rotation = makeCorrectionRotation(cameraCorrection);
        const Vector3 sensor = multiply(correction_rotation.value, nominal_sensor);
        if (!(sensor[2] > model.minimumDepthMeters) || !std::isfinite(sensor[0]) || !std::isfinite(sensor[1]) ||
            !std::isfinite(sensor[2]))
        {
            return false;
        }

        ProjectionEvaluation output;
        output.residualPixels[0] =
            model.principalSamplePixels + model.focalSamplePixels * sensor[0] / sensor[2] - observation.samplePixels;
        output.residualPixels[1] = model.focalLinePixels * sensor[1] / sensor[2] - model.detectorLinePixels;

        const std::array<double, 3> sample_gradient{
            {model.focalSamplePixels / sensor[2], 0.0, -model.focalSamplePixels * sensor[0] / (sensor[2] * sensor[2])}};
        const std::array<double, 3> line_gradient{
            {0.0, model.focalLinePixels / sensor[2], -model.focalLinePixels * sensor[1] / (sensor[2] * sensor[2])}};
        const Matrix3 point_rotation = multiply(correction_rotation.value, state.nominalPose.worldToSensorRotation);
        for (int row = 0; row < 2; ++row)
        {
            const auto& gradient = row == 0 ? sample_gradient : line_gradient;
            for (int column = 0; column < 3; ++column)
            {
                double derivative = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    derivative += gradient[axis] * point_rotation[axis * 3 + column];
                }
                output.pointJacobian[row * 3 + column] = derivative;
                output.cameraJacobian[row * 6 + column] = -derivative;
            }
            for (int angle = 0; angle < 3; ++angle)
            {
                const Vector3 angle_derivative = multiply(correction_rotation.derivatives[angle], nominal_sensor);
                double derivative = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    derivative += gradient[axis] * angle_derivative[axis];
                }
                output.cameraJacobian[row * 6 + angle + 3] = derivative;
            }
        }
        for (double value : output.residualPixels)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }
        *evaluation = output;
        return true;
    }

} // namespace plabundle::linescan

namespace plabundle::linescan::detail
{

    bool evaluateImageObservation(const Problem& problem,
                                  const ImageObservation& observation,
                                  const std::array<double, 6>& camera,
                                  const std::array<double, 3>& point,
                                  double imageSigmaPixels,
                                  double* residuals)
    {
        std::array<double, 2> raw{};
        if (!problem.cameraModels.empty())
        {
            ProjectionState state;
            ProjectionEvaluation evaluation;
            if (!prepareProjectionState(
                    problem.cameraModels[observation.cameraIndex], observation.linePixels, &state) ||
                !evaluateProjection(
                    problem.cameraModels[observation.cameraIndex], state, observation, camera, point, &evaluation))
            {
                return false;
            }
            raw = evaluation.residualPixels;
        }
        else if (!problem.projection || !problem.projection(observation, camera, point, &raw))
        {
            return false;
        }
        residuals[0] = raw[0] / imageSigmaPixels;
        residuals[1] = raw[1] / imageSigmaPixels;
        return std::isfinite(residuals[0]) && std::isfinite(residuals[1]);
    }

    double imageRms(const Problem& problem)
    {
        if (problem.imageObservations.empty())
        {
            return 0.0;
        }
        double squared = 0.0;
        for (const ImageObservation& observation : problem.imageObservations)
        {
            double residuals[2]{};
            if (!evaluateImageObservation(problem,
                                          observation,
                                          problem.cameraParameters[observation.cameraIndex],
                                          problem.tiePoints[observation.pointIndex],
                                          1.0,
                                          residuals))
            {
                return std::numeric_limits<double>::infinity();
            }
            squared += residuals[0] * residuals[0] + residuals[1] * residuals[1];
        }
        return std::sqrt(squared / (2.0 * problem.imageObservations.size()));
    }

    double laserRangeRms(const Problem& problem)
    {
        if (problem.laserObservations.empty())
        {
            return 0.0;
        }
        double squared = 0.0;
        for (const LaserObservation& observation : problem.laserObservations)
        {
            const auto& camera = problem.cameraParameters[observation.cameraIndex];
            const auto& point = problem.laserPoints[observation.laserPointIndex].refinedMeters;
            const std::array<double, 3> delta{{point[0] - observation.nominalSensorCenterMeters[0] - camera[0],
                                               point[1] - observation.nominalSensorCenterMeters[1] - camera[1],
                                               point[2] - observation.nominalSensorCenterMeters[2] - camera[2]}};
            const double residual = vectorNorm(delta) - observation.observedRangeMeters;
            squared += residual * residual;
        }
        return std::sqrt(squared / problem.laserObservations.size());
    }

} // namespace plabundle::linescan::detail
