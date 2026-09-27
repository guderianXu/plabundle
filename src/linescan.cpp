#include <plabundle/linescan.h>

#include "linescan_internal.h"

#include <plabundle/solver.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace plabundle::linescan
{
    namespace
    {

        void setError(std::string* output, const std::string& message)
        {
            if (output)
            {
                *output = message;
            }
        }

        bool finitePositive(double value)
        {
            return std::isfinite(value) && value > 0.0;
        }

        bool finiteVector(const std::array<double, 3>& value)
        {
            return std::all_of(value.begin(), value.end(), [](double element) { return std::isfinite(element); });
        }

        bool validOptions(const Options& options)
        {
            const bool supportedBackend =
                options.backend == Backend::Auto || options.backend == Backend::PlaMatrixCpu ||
                options.backend == Backend::PlaMatrixCuda || options.backend == Backend::PlaMatrixVulkan ||
                options.backend == Backend::PlaMatrixOpenCl;
            return supportedBackend && options.maximumIterations > 0 && options.threadCount >= 0 &&
                   options.plaMatrixDevice >= 0 && options.maxDenseSchurCameras > 0 &&
                   options.minPlaMatrixCudaCameras > 0 && options.minPlaMatrixCudaObservations > 0 &&
                   options.minPlaMatrixVulkanCameras > 0 && options.minPlaMatrixVulkanObservations > 0 &&
                   options.minPlaMatrixOpenClCameras > 0 && options.minPlaMatrixOpenClObservations > 0 &&
                   options.minPlaMatrixDenseCameras > 0 && options.minPlaMatrixCudaDenseObservations > 0 &&
                   options.minPlaMatrixVulkanDenseObservations > 0 && options.minPlaMatrixOpenClDenseObservations > 0 &&
                   finitePositive(options.imageSigmaPixels) && std::isfinite(options.imageHuberDeltaPixels) &&
                   options.imageHuberDeltaPixels >= 0.0 && finitePositive(options.cameraPositionSigmaMeters) &&
                   finitePositive(options.cameraAngleSigmaDegrees) && finitePositive(options.laserRangeWeight) &&
                   std::isfinite(options.laserRangeHuberDeltaSigma) && options.laserRangeHuberDeltaSigma >= 0.0 &&
                   finitePositive(options.finiteDifferencePointStepMeters) &&
                   finitePositive(options.finiteDifferencePositionStepMeters) &&
                   finitePositive(options.finiteDifferenceAngleStepRadians) &&
                   finitePositive(options.maximumCameraTranslationMeters) &&
                   finitePositive(options.maximumCameraAngleDegrees);
        }

        void
        addDiagnostic(Result* result, DiagnosticCode code, int cameraIndex, int pointIndex, const std::string& message)
        {
            result->diagnostics.push_back({code, cameraIndex, pointIndex, message});
            if (result->message.empty())
            {
                result->message = message;
            }
        }

        bool validProblem(const Problem& problem, const Options& options, Result* result)
        {
            if (problem.cameraParameters.size() < 2 || problem.tiePoints.empty() || problem.imageObservations.empty() ||
                (problem.cameraModels.empty() && !problem.projection) ||
                (!problem.cameraModels.empty() && problem.cameraModels.size() != problem.cameraParameters.size()))
            {
                return false;
            }
            for (const auto& camera : problem.cameraParameters)
            {
                if (!std::all_of(camera.begin(), camera.end(), [](double value) { return std::isfinite(value); }))
                {
                    return false;
                }
            }
            for (const auto& point : problem.tiePoints)
            {
                if (!finiteVector(point))
                {
                    return false;
                }
            }
            for (const auto& observation : problem.imageObservations)
            {
                if (observation.cameraIndex < 0 ||
                    static_cast<std::size_t>(observation.cameraIndex) >= problem.cameraParameters.size() ||
                    observation.pointIndex < 0 ||
                    static_cast<std::size_t>(observation.pointIndex) >= problem.tiePoints.size() ||
                    !std::isfinite(observation.samplePixels) || !std::isfinite(observation.linePixels))
                {
                    return false;
                }
            }
            std::vector<int> camera_observations(problem.cameraParameters.size(), 0);
            std::vector<int> point_observations(problem.tiePoints.size(), 0);
            for (const auto& observation : problem.imageObservations)
            {
                ++camera_observations[observation.cameraIndex];
                ++point_observations[observation.pointIndex];
                if (!problem.cameraModels.empty())
                {
                    ProjectionState state;
                    if (!prepareProjectionState(
                            problem.cameraModels[observation.cameraIndex], observation.linePixels, &state))
                    {
                        addDiagnostic(result,
                                      DiagnosticCode::InvalidCameraModel,
                                      observation.cameraIndex,
                                      observation.pointIndex,
                                      "line-scan camera timing or trajectory pose is invalid at an observed line");
                        return false;
                    }
                }
            }
            for (std::size_t index = 0; index < camera_observations.size(); ++index)
            {
                if (camera_observations[index] == 0)
                {
                    addDiagnostic(
                        result,
                        DiagnosticCode::UnobservedCamera,
                        static_cast<int>(index),
                        -1,
                        "line-scan camera correction is unobservable because the camera has no image observations");
                    return false;
                }
            }
            for (std::size_t index = 0; index < point_observations.size(); ++index)
            {
                if (point_observations[index] < 2)
                {
                    addDiagnostic(
                        result,
                        DiagnosticCode::SingleViewPoint,
                        -1,
                        static_cast<int>(index),
                        "line-scan tie point is unobservable because it has fewer than two image observations");
                    return false;
                }
            }
            for (const auto& point : problem.laserPoints)
            {
                if (!finiteVector(point.initialMeters) || !finiteVector(point.refinedMeters) ||
                    (point.mode != LaserPointMode::Fixed && point.mode != LaserPointMode::Constrained) ||
                    !std::all_of(point.sqrtInformation.begin(),
                                 point.sqrtInformation.end(),
                                 [](double value) { return std::isfinite(value); }))
                {
                    return false;
                }
                if (point.mode == LaserPointMode::Constrained &&
                    (!finitePositive(point.sqrtInformation[0]) || !finitePositive(point.sqrtInformation[4]) ||
                     !finitePositive(point.sqrtInformation[8])))
                {
                    return false;
                }
            }
            for (const auto& observation : problem.laserObservations)
            {
                if (observation.cameraIndex < 0 ||
                    static_cast<std::size_t>(observation.cameraIndex) >= problem.cameraParameters.size() ||
                    observation.laserPointIndex < 0 ||
                    static_cast<std::size_t>(observation.laserPointIndex) >= problem.laserPoints.size() ||
                    !finiteVector(observation.nominalSensorCenterMeters) ||
                    !finitePositive(observation.observedRangeMeters) || !finitePositive(observation.sigmaMeters))
                {
                    return false;
                }
            }
            return !options.enableLaserRangeConstraints || !problem.laserObservations.empty();
        }

        Backend selectBackend(const Options& options, const Problem& problem)
        {
            if (options.backend != Backend::Auto)
            {
                return options.backend;
            }
            plabundle::Options policy;
            policy.minPlaMatrixCudaCameras = options.minPlaMatrixCudaCameras;
            policy.minPlaMatrixCudaObservations = options.minPlaMatrixCudaObservations;
            policy.minPlaMatrixVulkanCameras = options.minPlaMatrixVulkanCameras;
            policy.minPlaMatrixVulkanObservations = options.minPlaMatrixVulkanObservations;
            policy.minPlaMatrixOpenClCameras = options.minPlaMatrixOpenClCameras;
            policy.minPlaMatrixOpenClObservations = options.minPlaMatrixOpenClObservations;
            policy.minPlaMatrixDenseCameras = options.minPlaMatrixDenseCameras;
            policy.minPlaMatrixCudaDenseObservations = options.minPlaMatrixCudaDenseObservations;
            policy.minPlaMatrixVulkanDenseObservations = options.minPlaMatrixVulkanDenseObservations;
            policy.minPlaMatrixOpenClDenseObservations = options.minPlaMatrixOpenClDenseObservations;
            ProblemStats stats;
            stats.cameraCount = static_cast<int>(problem.cameraParameters.size());
            stats.observationCount = static_cast<int>(problem.imageObservations.size());
            for (Backend backend : {Backend::PlaMatrixCuda, Backend::PlaMatrixVulkan, Backend::PlaMatrixOpenCl})
            {
                if (Solver::autoBackendMeetsScaleThreshold(backend, stats, policy) &&
                    Solver::isBackendAvailable(backend))
                {
                    return backend;
                }
            }
            return Backend::PlaMatrixCpu;
        }

        bool qualityGate(const Problem& problem, const Options& options, Result* result, std::string* errorMessage)
        {
            result->refinedImageRmsPixels = detail::imageRms(problem);
            result->refinedLaserRangeRmsMeters = detail::laserRangeRms(problem);
            if (!std::isfinite(result->refinedImageRmsPixels) || !std::isfinite(result->refinedLaserRangeRmsMeters) ||
                result->refinedImageRmsPixels >
                    result->initialImageRmsPixels + std::max(0.25, 0.5 * result->initialImageRmsPixels))
            {
                setError(errorMessage, "line-scan BA quality gate rejected non-finite or degraded residuals");
                return false;
            }
            for (const auto& camera : problem.cameraParameters)
            {
                const double translation = std::hypot(std::hypot(camera[0], camera[1]), camera[2]);
                const double angle = std::hypot(std::hypot(camera[3], camera[4]), camera[5]) * 180.0 / std::acos(-1.0);
                if (!std::isfinite(translation) || !std::isfinite(angle) ||
                    translation > options.maximumCameraTranslationMeters || angle > options.maximumCameraAngleDegrees)
                {
                    setError(errorMessage, "line-scan BA quality gate rejected an excessive camera correction");
                    return false;
                }
            }
            for (const auto& point : problem.tiePoints)
            {
                if (!finiteVector(point))
                {
                    setError(errorMessage, "line-scan BA produced a non-finite control point");
                    return false;
                }
            }
            return true;
        }

    } // namespace

    Result solve(const Problem& problem, const Options& options)
    {
        Result result;
        result.requestedBackend = options.backend;
        if (!validOptions(options) || !validProblem(problem, options, &result))
        {
            result.terminationType = "INVALID_INPUT";
            if (result.message.empty())
            {
                result.message = "invalid line-scan BA problem or options";
            }
            result.backendMessage = result.message;
            return result;
        }
        Problem candidate = problem;
        result.initialImageRmsPixels = detail::imageRms(candidate);
        result.initialLaserRangeRmsMeters = detail::laserRangeRms(candidate);
        if (!std::isfinite(result.initialImageRmsPixels) || !std::isfinite(result.initialLaserRangeRmsMeters))
        {
            result.terminationType = "INVALID_INPUT";
            result.message = "line-scan BA initial residual is non-finite";
            result.backendMessage = result.message;
            addDiagnostic(
                &result,
                DiagnosticCode::DegenerateProjection,
                -1,
                -1,
                "line-scan projection is degenerate at the initial state (invalid depth or detector geometry)");
            return result;
        }
        Options solverOptions = options;
        solverOptions.backend = selectBackend(options, candidate);
        result.usedBackend = solverOptions.backend;
        std::string error_message;
        bool solved = detail::solvePlaMatrix(&candidate, solverOptions, &result, &error_message);
        if (!solved && options.allowBackendFallback && result.terminationType != "CANCELLED" &&
            (solverOptions.backend == Backend::PlaMatrixCuda || solverOptions.backend == Backend::PlaMatrixVulkan ||
             solverOptions.backend == Backend::PlaMatrixOpenCl))
        {
            const std::string accelerator_error = error_message;
            candidate = problem;
            solverOptions.backend = Backend::PlaMatrixCpu;
            result.backendFallback = true;
            result.usedBackend = Backend::PlaMatrixCpu;
            solved = detail::solvePlaMatrix(&candidate, solverOptions, &result, &error_message);
            if (solved)
            {
                result.backendMessage =
                    "PlaMatrix line-scan BA failed: " + accelerator_error + "; fell back to plamatrix_cpu";
            }
        }
        if (!solved || !qualityGate(candidate, options, &result, &error_message))
        {
            if (result.message.empty())
            {
                result.message = error_message.empty() ? result.backendMessage : error_message;
            }
            return result;
        }
        if (result.backendMessage.empty())
        {
            result.backendMessage = std::string("line-scan BA used ") + backendName(result.usedBackend);
        }
        result.refinedCameraParameters = std::move(candidate.cameraParameters);
        result.refinedTiePoints = std::move(candidate.tiePoints);
        result.refinedLaserPoints.reserve(candidate.laserPoints.size());
        for (const LaserPoint& point : candidate.laserPoints)
        {
            result.refinedLaserPoints.push_back(point.refinedMeters);
        }
        result.success = true;
        return result;
    }

    bool solve(Problem* problem, const Options& options, Result* result, std::string* errorMessage)
    {
        if (!result)
        {
            setError(errorMessage, "output line-scan BA result pointer is null");
            return false;
        }
        if (!problem)
        {
            *result = Result{};
            result->requestedBackend = options.backend;
            result->terminationType = "INVALID_INPUT";
            result->message = "input line-scan BA problem pointer is null";
            result->backendMessage = result->message;
            setError(errorMessage, result->message);
            return false;
        }

        *result = solve(*problem, options);
        if (!result->usable())
        {
            setError(errorMessage, result->message.empty() ? result->backendMessage : result->message);
            return false;
        }

        problem->cameraParameters = result->refinedCameraParameters;
        problem->tiePoints = result->refinedTiePoints;
        for (std::size_t index = 0; index < problem->laserPoints.size(); ++index)
        {
            problem->laserPoints[index].refinedMeters = result->refinedLaserPoints[index];
        }
        return true;
    }

} // namespace plabundle::linescan
