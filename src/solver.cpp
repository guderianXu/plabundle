#include <plabundle/solver.h>

#include "internal/BundleAdjustPlaMatrix.h"
#include "internal/BundleAdjustPlaMatrixRuntime.h"
#include "internal/BundleAdjustQuality.h"
#include "internal/BundleAdjustValidation.h"
#include "internal/Conversion.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace plabundle
{
    namespace
    {
        Result makeFailure(const Problem& problem,
                           Backend requestedBackend,
                           Backend usedBackend,
                           SolveStatus status,
                           std::string message,
                           std::string selectionReason = {})
        {
            Result result;
            result.requestedBackend = requestedBackend;
            result.usedBackend = usedBackend;
            result.status = status;
            result.backendMessage = std::move(message);
            result.backendSelectionReason = std::move(selectionReason);
            result.observationCount = summarizeProblem(problem).observationCount;
            result.quality.totalTracks = static_cast<int>(problem.tracks.size());
            result.refinedCameras = problem.cameras;
            result.points.resize(problem.tracks.size());
            for (std::size_t index = 0; index < problem.tracks.size(); ++index)
            {
                result.points[index].point = problem.tracks[index].initialPoint;
            }
            return result;
        }

        bool resultFailsQualityGate(const Result& result, const Options& options, std::string* message)
        {
            if (!options.enableBackendQualityGate)
            {
                return false;
            }
            if (!result.solutionUsable)
            {
                if (message)
                {
                    *message = "quality gate rejected a non-usable solution";
                }
                return true;
            }

            const QualitySummary& quality = result.quality;
            if (quality.totalTracks > 0 && !std::isfinite(quality.meanRmsAfter))
            {
                if (message)
                {
                    *message = "quality gate rejected a non-finite final RMS";
                }
                return true;
            }
            if (quality.totalTracks > 0 && quality.validTrackRatio < std::max(0.0, options.minAcceptedValidTrackRatio))
            {
                if (message)
                {
                    *message = "quality gate rejected a low valid-track ratio";
                }
                return true;
            }

            const double max_growth = std::max(0.0, options.maxAcceptedRmsGrowth);
            if (quality.totalTracks > 0 && max_growth > 0.0 && std::isfinite(quality.meanRmsBefore))
            {
                if (quality.meanRmsBefore > 1.0e-12 && quality.meanRmsAfter > quality.meanRmsBefore * max_growth)
                {
                    if (message)
                    {
                        *message = "quality gate rejected excessive reprojection RMS growth";
                    }
                    return true;
                }
                if (quality.meanRmsBefore <= 1.0e-12 && quality.meanRmsAfter > 1.0e-9)
                {
                    if (message)
                    {
                        *message = "quality gate rejected growth from a zero-residual input";
                    }
                    return true;
                }
            }

            const double max_constraint_growth = std::max(1.0, options.maxAcceptedConstraintRmsGrowth);
            return !internal::constraintRmsPassesQualityGate(quality.laserConstraintCount,
                                                             quality.laserRmsBeforeMeters,
                                                             quality.laserRmsAfterMeters,
                                                             max_constraint_growth,
                                                             "laser-plane constraints",
                                                             message) ||
                   !internal::constraintRmsPassesQualityGate(quality.laserRangeConstraintCount,
                                                             quality.laserRangeRmsBeforeMeters,
                                                             quality.laserRangeRmsAfterMeters,
                                                             max_constraint_growth,
                                                             "laser-range constraints",
                                                             message) ||
                   !internal::constraintRmsPassesQualityGate(quality.controlPointConstraintCount,
                                                             quality.controlPointRmsBeforeMeters,
                                                             quality.controlPointRmsAfterMeters,
                                                             max_constraint_growth,
                                                             "control-point constraints",
                                                             message) ||
                   !internal::constraintRmsPassesQualityGate(quality.scaleBarConstraintCount,
                                                             quality.scaleBarRmsBeforeMeters,
                                                             quality.scaleBarRmsAfterMeters,
                                                             max_constraint_growth,
                                                             "scale-bar constraints",
                                                             message);
        }

        bool hasSoftConstraints(const Problem& problem)
        {
            const bool has_track_constraints =
                std::any_of(problem.tracks.begin(),
                            problem.tracks.end(),
                            [](const Track& track)
                            { return !track.laserPlaneConstraints.empty() || !track.controlPointConstraints.empty(); });
            const bool has_pose_prior = std::any_of(problem.cameraPosePriors.begin(),
                                                    problem.cameraPosePriors.end(),
                                                    [](const auto& prior) { return prior.has_value(); });
            return has_track_constraints || !problem.laserRangeConstraints.empty() ||
                   !problem.scaleBarConstraints.empty() || problem.cameraPlaneConstraint.has_value() || has_pose_prior;
        }

        bool needsJointSolver(const Problem& problem, const Options& options)
        {
            const bool refine_extended_intrinsics =
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::FocalAspectRatio) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::PrincipalPointX) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::PrincipalPointY) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::RadialK1) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::RadialK2) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::RadialK3) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::TangentialP1) ||
                sharedIntrinsicParameterEnabled(options, IntrinsicParameter::TangentialP2);
            return options.refineCameraPose ||
                   sharedIntrinsicParameterEnabled(options, IntrinsicParameter::FocalLength) ||
                   refine_extended_intrinsics || hasSoftConstraints(problem);
        }

        std::string joinedMessage(const std::string& prefix, const std::string& detail)
        {
            if (prefix.empty())
            {
                return detail;
            }
            if (detail.empty())
            {
                return prefix;
            }
            return prefix + "; " + detail;
        }

        void accumulateTiming(Result* target, const Result& source)
        {
            target->timing.setupSeconds += source.timing.setupSeconds;
            target->timing.solveSeconds += source.timing.solveSeconds;
            target->timing.postprocessSeconds += source.timing.postprocessSeconds;
            target->timing.totalSeconds += source.timing.totalSeconds;
        }

        Result runConcreteBackend(const Problem& problem,
                                  const Options& options,
                                  Backend requestedBackend,
                                  Backend usedBackend,
                                  const std::string& selectionReason)
        {
            const std::vector<internal::CameraState> cameras = internal::makeCameraStates(problem.cameras);
            const internal::BAOptions solver_options = internal::makeSolverOptions(problem, options, usedBackend);
            internal::BAOptions normalized_options;
            const internal::BundleAdjustValidationResult validation = internal::validateAndNormalizeBundleAdjustOptions(
                cameras, problem.tracks, solver_options, &normalized_options);
            if (!validation.ok)
            {
                return makeFailure(
                    problem, requestedBackend, usedBackend, validation.status, validation.message, selectionReason);
            }

            const internal::BAResult solver_result =
                internal::optimizePointsWithPlaMatrix(cameras, problem.tracks, normalized_options);
            return internal::makePublicResult(solver_result, requestedBackend, usedBackend, selectionReason);
        }

        Result runCpuFallback(const Problem& problem,
                              const Options& options,
                              Backend requestedBackend,
                              const std::string& reason,
                              const Result* previousAttempt = nullptr)
        {
            Options cpu_options = options;
            cpu_options.backend = Backend::PlaMatrixCpu;
            cpu_options.allowBackendFallback = false;
            Result fallback = runConcreteBackend(problem, cpu_options, requestedBackend, Backend::PlaMatrixCpu, reason);
            fallback.backendFallback = requestedBackend != Backend::PlaMatrixCpu;
            fallback.backendSelectionReason = reason;
            fallback.backendMessage = joinedMessage(reason, fallback.backendMessage);
            if (previousAttempt)
            {
                accumulateTiming(&fallback, *previousAttempt);
            }
            return fallback;
        }

        void rejectByQualityGate(Result* result, const std::string& selectionReason, const std::string& message)
        {
            result->qualityGateRejected = true;
            result->qualityGateMessage = message;
            result->solutionUsable = false;
            if (result->status == SolveStatus::Success || result->status == SolveStatus::NoConvergence)
            {
                result->status = SolveStatus::NumericalFailure;
            }
            result->backendFallback = false;
            result->backendSelectionReason = selectionReason;
            result->backendMessage = joinedMessage(selectionReason, message);
        }

    } // namespace

    Result Solver::solve(const Problem& problem, const Options& options) const
    {
        const Backend initial_backend = options.backend == Backend::Auto ? Backend::PlaMatrixCpu : options.backend;

        std::string error;
        if (!validateOptions(options, &error))
        {
            return makeFailure(problem, options.backend, initial_backend, SolveStatus::InvalidInput, error);
        }
        if (!validateProblem(problem, &error))
        {
            return makeFailure(problem, options.backend, initial_backend, SolveStatus::InvalidInput, error);
        }
        if (options.stopToken.stop_requested() ||
            (options.cancelFlag && options.cancelFlag->load(std::memory_order_relaxed)))
        {
            return makeFailure(
                problem, options.backend, initial_backend, SolveStatus::Cancelled, "solve cancelled before execution");
        }

        try
        {
            if (options.backend == Backend::Auto)
            {
                const BackendDecision decision = decideBackendForProblem(problem, options);
                Options selected_options = options;
                selected_options.backend = decision.backend;
                selected_options.allowBackendFallback = false;
                Result candidate =
                    runConcreteBackend(problem, selected_options, Backend::Auto, decision.backend, decision.reason);
                candidate.requestedBackend = Backend::Auto;
                if (candidate.status == SolveStatus::Cancelled)
                {
                    candidate.backendSelectionReason = decision.reason + "; cancelled_no_fallback";
                    return candidate;
                }

                std::string quality_message;
                if (resultFailsQualityGate(candidate, options, &quality_message))
                {
                    if (decision.backend == Backend::PlaMatrixCpu)
                    {
                        rejectByQualityGate(
                            &candidate, decision.reason + "; cpu_quality_gate_rejected", quality_message);
                        return candidate;
                    }
                    const std::string fallback_reason =
                        decision.reason + "; accelerated_candidate_quality_gate_rejected; fallback_to_plamatrix_cpu";
                    Result fallback = runCpuFallback(problem, options, Backend::Auto, fallback_reason, &candidate);
                    fallback.qualityGateRejected = true;
                    fallback.qualityGateMessage = quality_message;
                    fallback.backendMessage = joinedMessage(fallback.backendMessage, quality_message);
                    return fallback;
                }

                candidate.backendSelectionReason = decision.reason + "; quality_gate_passed";
                candidate.backendMessage = joinedMessage(candidate.backendSelectionReason, candidate.backendMessage);
                return candidate;
            }

            const Backend requested_backend = options.backend;
            std::string unavailable_message;
            if (!internal::isPlaMatrixBackendAvailable(
                    requested_backend, options.plaMatrixDevice, &unavailable_message))
            {
                if (options.allowBackendFallback && requested_backend != Backend::PlaMatrixCpu)
                {
                    return runCpuFallback(
                        problem, options, requested_backend, unavailable_message + "; fallback_to_plamatrix_cpu");
                }
                return makeFailure(problem,
                                   requested_backend,
                                   requested_backend,
                                   SolveStatus::BackendUnavailable,
                                   unavailable_message,
                                   "explicit_backend");
            }

            Result result =
                runConcreteBackend(problem, options, requested_backend, requested_backend, "explicit_backend");
            if (!result.solutionUsable && result.status != SolveStatus::Cancelled && options.allowBackendFallback &&
                requested_backend != Backend::PlaMatrixCpu)
            {
                return runCpuFallback(problem,
                                      options,
                                      requested_backend,
                                      joinedMessage(result.backendMessage, "fallback_to_plamatrix_cpu"),
                                      &result);
            }
            return result;
        }
        catch (const std::bad_alloc&)
        {
            throw;
        }
        catch (const std::exception& exception)
        {
            return makeFailure(problem,
                               options.backend,
                               initial_backend,
                               SolveStatus::NumericalFailure,
                               std::string("PlaMatrix solve failed: ") + exception.what());
        }
    }

    bool Solver::isBackendAvailable(Backend backend, int deviceIndex) noexcept
    {
        if (backend == Backend::Auto)
        {
            return true;
        }
        try
        {
            return internal::isPlaMatrixBackendAvailable(backend, deviceIndex);
        }
        catch (...)
        {
            return false;
        }
    }

    BackendCapabilities Solver::backendCapabilities(Backend backend) noexcept
    {
        switch (backend)
        {
        case Backend::Auto:
        case Backend::PlaMatrixCpu:
        case Backend::PlaMatrixCuda:
        case Backend::PlaMatrixOpenCl:
            return {true, true, true, true, true, true, true, true};
        default:
            return {};
        }
    }

    bool
    Solver::autoBackendMeetsScaleThreshold(Backend backend, const ProblemStats& stats, const Options& options) noexcept
    {
        const int camera_count = std::max(0, stats.cameraCount);
        const int observation_count = std::max(0, stats.observationCount);
        const int dense_camera_threshold = std::max(1, options.minPlaMatrixDenseCameras);
        if (backend == Backend::PlaMatrixCuda)
        {
            const bool regular_scale = camera_count >= std::max(1, options.minPlaMatrixCudaCameras) &&
                                       observation_count >= std::max(1, options.minPlaMatrixCudaObservations);
            const bool dense_scale = camera_count >= dense_camera_threshold &&
                                     observation_count >= std::max(1, options.minPlaMatrixCudaDenseObservations);
            return regular_scale || dense_scale;
        }
        if (backend == Backend::PlaMatrixOpenCl)
        {
            const bool regular_scale = camera_count >= std::max(1, options.minPlaMatrixOpenClCameras) &&
                                       observation_count >= std::max(1, options.minPlaMatrixOpenClObservations);
            const bool dense_scale = camera_count >= dense_camera_threshold &&
                                     observation_count >= std::max(1, options.minPlaMatrixOpenClDenseObservations);
            return regular_scale || dense_scale;
        }
        return false;
    }

    BackendDecision Solver::decideBackendForProblem(const Problem& problem, const Options& options)
    {
        if (options.backend != Backend::Auto)
        {
            return {options.backend, "explicit_backend"};
        }
        if (!needsJointSolver(problem, options))
        {
            return {Backend::PlaMatrixCpu, "point_only_uses_reference_ba"};
        }

        const ProblemStats stats = summarizeProblem(problem);
        if (autoBackendMeetsScaleThreshold(Backend::PlaMatrixCuda, stats, options) &&
            isBackendAvailable(Backend::PlaMatrixCuda, options.plaMatrixDevice))
        {
            const bool regular_scale = stats.cameraCount >= std::max(1, options.minPlaMatrixCudaCameras) &&
                                       stats.observationCount >= std::max(1, options.minPlaMatrixCudaObservations);
            return {Backend::PlaMatrixCuda,
                    regular_scale ? "large_joint_problem_uses_plamatrix_cuda"
                                  : "dense_joint_problem_uses_plamatrix_cuda"};
        }
        if (autoBackendMeetsScaleThreshold(Backend::PlaMatrixOpenCl, stats, options) &&
            isBackendAvailable(Backend::PlaMatrixOpenCl, options.plaMatrixDevice))
        {
            const bool regular_scale = stats.cameraCount >= std::max(1, options.minPlaMatrixOpenClCameras) &&
                                       stats.observationCount >= std::max(1, options.minPlaMatrixOpenClObservations);
            return {Backend::PlaMatrixOpenCl,
                    regular_scale ? "large_joint_problem_uses_plamatrix_opencl"
                                  : "dense_joint_problem_uses_plamatrix_opencl"};
        }
        return {Backend::PlaMatrixCpu,
                hasSoftConstraints(problem) ? "constraint_problem_uses_plamatrix_cpu"
                                            : "joint_problem_uses_plamatrix_cpu"};
    }

    Backend Solver::selectBackendForProblem(const Problem& problem, const Options& options)
    {
        return decideBackendForProblem(problem, options).backend;
    }

} // namespace plabundle
