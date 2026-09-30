#include <plabundle/solver.h>

#include "control_point_internal.h"
#include "internal/BundleAdjustPlaMatrix.h"
#include "internal/BundleAdjustPlaMatrixRuntime.h"
#include "internal/BundleAdjustQuality.h"
#include "internal/BundleAdjustValidation.h"
#include "internal/Conversion.h"

#include <placamera/rig_topology.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace plabundle
{
    struct SolverWorkspace::Impl
    {
        plamatrix::internal::SchurComplementSolverWorkspace<double> linearWorkspace;
    };

    SolverWorkspace::SolverWorkspace() : _impl(std::make_unique<Impl>()) {}
    SolverWorkspace::~SolverWorkspace() = default;
    SolverWorkspace::SolverWorkspace(SolverWorkspace&&) noexcept = default;
    SolverWorkspace& SolverWorkspace::operator=(SolverWorkspace&&) noexcept = default;

    void SolverWorkspace::clear() noexcept
    {
        if (_impl)
        {
            _impl->linearWorkspace.clear();
        }
    }

    namespace
    {
        struct ConstraintRmsUncertainty
        {
            double laserPlaneMeters = 0.0;
            double laserRangeMeters = 0.0;
            double controlPointMeters = 0.0;
            double scaleBarMeters = 0.0;
        };

        ConstraintRmsUncertainty constraintRmsUncertainty(const Problem& problem, const Options& options)
        {
            double laser_plane_variance = 0.0;
            double laser_range_variance = 0.0;
            double control_point_variance = 0.0;
            double scale_bar_variance = 0.0;
            std::size_t laser_plane_count = 0;
            std::size_t laser_range_count = 0;
            std::size_t control_point_count = 0;
            std::size_t scale_bar_count = 0;
            for (const Track& track : problem.tracks)
            {
                for (const LaserPlaneConstraint& constraint : track.laserPlaneConstraints)
                {
                    laser_plane_variance += 1.0 / (options.laserPlaneWeight * constraint.weight);
                    ++laser_plane_count;
                }
                for (const ControlPointConstraint& constraint : track.controlPointConstraints)
                {
                    double rms_uncertainty = 0.0;
                    if (internal::controlPointRmsUncertaintyMeters(constraint, &rms_uncertainty))
                    {
                        control_point_variance +=
                            rms_uncertainty * rms_uncertainty / (options.controlPointWeight * constraint.weight);
                        ++control_point_count;
                    }
                }
            }
            for (const LaserRangeConstraint& constraint : problem.laserRangeConstraints)
            {
                laser_range_variance += constraint.sigmaRangeMeters * constraint.sigmaRangeMeters /
                                        (options.laserRangeWeight * constraint.weight);
                ++laser_range_count;
            }
            for (const ScaleBarConstraint& constraint : problem.scaleBarConstraints)
            {
                scale_bar_variance +=
                    constraint.sigmaMeters * constraint.sigmaMeters / (options.scaleBarWeight * constraint.weight);
                ++scale_bar_count;
            }
            const auto rms = [](double variance, std::size_t count)
            { return count > 0 ? std::sqrt(variance / static_cast<double>(count)) : 0.0; };
            return {rms(laser_plane_variance, laser_plane_count),
                    rms(laser_range_variance, laser_range_count),
                    rms(control_point_variance, control_point_count),
                    rms(scale_bar_variance, scale_bar_count)};
        }

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
            result.refinedRig = problem.rig;
            result.points.resize(problem.tracks.size());
            for (std::size_t index = 0; index < problem.tracks.size(); ++index)
            {
                result.points[index].point = problem.tracks[index].initialPoint;
            }
            return result;
        }

        bool resultFailsQualityGate(const Problem& problem,
                                    const Result& result,
                                    const Options& options,
                                    std::string* message)
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
            const ConstraintRmsUncertainty uncertainty = constraintRmsUncertainty(problem, options);
            return !internal::constraintRmsPassesQualityGate(quality.laserConstraintCount,
                                                             quality.laserRmsBeforeMeters,
                                                             quality.laserRmsAfterMeters,
                                                             max_constraint_growth,
                                                             uncertainty.laserPlaneMeters,
                                                             "laser-plane constraints",
                                                             message) ||
                   !internal::constraintRmsPassesQualityGate(quality.laserRangeConstraintCount,
                                                             quality.laserRangeRmsBeforeMeters,
                                                             quality.laserRangeRmsAfterMeters,
                                                             max_constraint_growth,
                                                             uncertainty.laserRangeMeters,
                                                             "laser-range constraints",
                                                             message) ||
                   !internal::constraintRmsPassesQualityGate(quality.controlPointConstraintCount,
                                                             quality.controlPointRmsBeforeMeters,
                                                             quality.controlPointRmsAfterMeters,
                                                             max_constraint_growth,
                                                             uncertainty.controlPointMeters,
                                                             "control-point constraints",
                                                             message) ||
                   !internal::constraintRmsPassesQualityGate(quality.scaleBarConstraintCount,
                                                             quality.scaleBarRmsBeforeMeters,
                                                             quality.scaleBarRmsAfterMeters,
                                                             max_constraint_growth,
                                                             uncertainty.scaleBarMeters,
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
            bool refine_extended_intrinsics = false;
            for (std::size_t parameter = 1; parameter < kIntrinsicParameterCount; ++parameter)
            {
                refine_extended_intrinsics =
                    refine_extended_intrinsics ||
                    sharedIntrinsicParameterEnabled(options, static_cast<IntrinsicParameter>(parameter));
            }
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

        void rejectByQualityGate(Result* result, const std::string& selectionReason, const std::string& message)
        {
            const std::string solver_message = result->backendMessage;
            result->qualityGateRejected = true;
            result->qualityGateMessage = message;
            result->solutionUsable = false;
            if (result->status == SolveStatus::Success || result->status == SolveStatus::NoConvergence)
            {
                result->status = SolveStatus::NumericalFailure;
            }
            result->backendSelectionReason = selectionReason;
            result->backendMessage = joinedMessage(joinedMessage(selectionReason, solver_message), message);
        }

        Result runConcreteBackend(const Problem& problem,
                                  const Options& options,
                                  Backend requestedBackend,
                                  Backend usedBackend,
                                  const std::string& selectionReason,
                                  plamatrix::internal::SchurComplementSolverWorkspace<double>* linear_workspace)
        {
            const auto composed_cameras = placamera::composeRigCameras(problem.cameras, problem.rig);
            if (!composed_cameras)
            {
                return makeFailure(
                    problem, requestedBackend, usedBackend, SolveStatus::InvalidInput, composed_cameras.message(), selectionReason);
            }
            const std::vector<internal::CameraState> cameras = internal::makeCameraStates(composed_cameras.value());
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
                internal::optimizePointsWithPlaMatrix(cameras, problem.tracks, normalized_options, linear_workspace);
            return internal::makePublicResult(solver_result, problem, requestedBackend, usedBackend, selectionReason);
        }

        Result runCpuFallback(const Problem& problem,
                              const Options& options,
                              Backend requestedBackend,
                              const std::string& reason,
                              plamatrix::internal::SchurComplementSolverWorkspace<double>* linear_workspace,
                              const Result* previousAttempt = nullptr)
        {
            Options cpu_options = options;
            cpu_options.backend = Backend::PlaMatrixCpu;
            cpu_options.allowBackendFallback = false;
            Result fallback = runConcreteBackend(
                problem, cpu_options, requestedBackend, Backend::PlaMatrixCpu, reason, linear_workspace);
            fallback.backendFallback = requestedBackend != Backend::PlaMatrixCpu;
            fallback.backendSelectionReason = reason;
            fallback.backendMessage = joinedMessage(reason, fallback.backendMessage);
            if (previousAttempt)
            {
                accumulateTiming(&fallback, *previousAttempt);
            }
            return fallback;
        }

        Result runQualityCheckedCpuFallback(const Problem& problem,
                                            const Options& options,
                                            Backend requestedBackend,
                                            const std::string& reason,
                                            plamatrix::internal::SchurComplementSolverWorkspace<double>* linear_workspace,
                                            const Result* previousAttempt = nullptr,
                                            const std::string& previousQualityMessage = {})
        {
            Result fallback = runCpuFallback(
                problem, options, requestedBackend, reason, linear_workspace, previousAttempt);
            if (fallback.status == SolveStatus::Cancelled)
            {
                fallback.backendSelectionReason = reason + "; cpu_fallback_cancelled";
                return fallback;
            }
            std::string fallback_quality_message;
            if (resultFailsQualityGate(problem, fallback, options, &fallback_quality_message))
            {
                const std::string combined_message =
                    previousQualityMessage.empty()
                        ? fallback_quality_message
                        : joinedMessage("candidate quality rejection: " + previousQualityMessage,
                                        "CPU fallback quality rejection: " + fallback_quality_message);
                rejectByQualityGate(&fallback, reason + "; cpu_fallback_quality_gate_rejected", combined_message);
                return fallback;
            }
            fallback.backendSelectionReason =
                reason + (options.enableBackendQualityGate ? "; cpu_fallback_quality_gate_passed"
                                                           : "; cpu_fallback_quality_gate_disabled");
            if (!previousQualityMessage.empty())
            {
                fallback.qualityGateRejected = true;
                fallback.qualityGateMessage = previousQualityMessage;
                fallback.backendMessage = joinedMessage(fallback.backendMessage, previousQualityMessage);
            }
            return fallback;
        }

    } // namespace

    Result Solver::solve(const Problem& problem) const
    {
        return solve(problem, SolveOptions{});
    }

    Result Solver::solve(const Problem& problem, const SolveOptions& options) const
    {
        return solve(problem, makeCompatibilityOptions(options));
    }

    Result Solver::solve(const Problem& problem, const Options& options) const
    {
        SolverWorkspace workspace;
        return solve(problem, options, workspace);
    }

    Result Solver::solve(const Problem& problem, const SolveOptions& options, SolverWorkspace& workspace) const
    {
        return solve(problem, makeCompatibilityOptions(options), workspace);
    }

    Result Solver::solve(const Problem& problem, const Options& options, SolverWorkspace& workspace) const
    {
        if (!workspace._impl)
        {
            workspace._impl = std::make_unique<SolverWorkspace::Impl>();
        }
        auto* linear_workspace = &workspace._impl->linearWorkspace;
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
                    runConcreteBackend(problem, selected_options, Backend::Auto, decision.backend, decision.reason,
                                       linear_workspace);
                candidate.requestedBackend = Backend::Auto;
                if (candidate.status == SolveStatus::Cancelled)
                {
                    candidate.backendSelectionReason = decision.reason + "; cancelled_no_fallback";
                    return candidate;
                }

                std::string quality_message;
                if (resultFailsQualityGate(problem, candidate, options, &quality_message))
                {
                    if (decision.backend == Backend::PlaMatrixCpu)
                    {
                        rejectByQualityGate(
                            &candidate, decision.reason + "; cpu_quality_gate_rejected", quality_message);
                        return candidate;
                    }
                    const std::string fallback_reason =
                        decision.reason + "; accelerated_candidate_quality_gate_rejected; fallback_to_plamatrix_cpu";
                    return runQualityCheckedCpuFallback(
                        problem, options, Backend::Auto, fallback_reason, linear_workspace, &candidate, quality_message);
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
                    return runQualityCheckedCpuFallback(
                        problem, options, requested_backend, unavailable_message + "; fallback_to_plamatrix_cpu",
                        linear_workspace);
                }
                return makeFailure(problem,
                                   requested_backend,
                                   requested_backend,
                                   SolveStatus::BackendUnavailable,
                                   unavailable_message,
                                   "explicit_backend");
            }

            Result result =
                runConcreteBackend(problem, options, requested_backend, requested_backend, "explicit_backend",
                                   linear_workspace);
            if (result.status == SolveStatus::Cancelled)
            {
                result.backendSelectionReason = "explicit_backend; cancelled_no_fallback";
                return result;
            }

            std::string quality_message;
            if (resultFailsQualityGate(problem, result, options, &quality_message))
            {
                if (options.allowBackendFallback && requested_backend != Backend::PlaMatrixCpu)
                {
                    const std::string fallback_reason =
                        "explicit_backend; accelerated_candidate_quality_gate_rejected; fallback_to_plamatrix_cpu";
                    return runQualityCheckedCpuFallback(
                        problem, options, requested_backend, fallback_reason, linear_workspace, &result, quality_message);
                }
                rejectByQualityGate(&result, "explicit_backend; quality_gate_rejected", quality_message);
                return result;
            }
            if (!result.solutionUsable && result.status != SolveStatus::Cancelled && options.allowBackendFallback &&
                requested_backend != Backend::PlaMatrixCpu)
            {
                return runQualityCheckedCpuFallback(problem,
                                                    options,
                                                    requested_backend,
                                                    joinedMessage(result.backendMessage, "fallback_to_plamatrix_cpu"),
                                                    linear_workspace,
                                                    &result);
            }
            result.backendSelectionReason = options.enableBackendQualityGate
                                                ? "explicit_backend; quality_gate_passed"
                                                : "explicit_backend; quality_gate_disabled";
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
        case Backend::PlaMatrixVulkan:
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
        if (backend == Backend::PlaMatrixVulkan)
        {
            const bool regular_scale = camera_count >= std::max(1, options.minPlaMatrixVulkanCameras) &&
                                       observation_count >= std::max(1, options.minPlaMatrixVulkanObservations);
            const bool dense_scale = camera_count >= dense_camera_threshold &&
                                     observation_count >= std::max(1, options.minPlaMatrixVulkanDenseObservations);
            return regular_scale || dense_scale;
        }
        return false;
    }

    bool Solver::autoBackendMeetsScaleThreshold(Backend backend,
                                                const ProblemStats& stats,
                                                const SolveOptions& options) noexcept
    {
        try
        {
            return autoBackendMeetsScaleThreshold(backend, stats, makeCompatibilityOptions(options));
        }
        catch (...)
        {
            return false;
        }
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
        if (autoBackendMeetsScaleThreshold(Backend::PlaMatrixVulkan, stats, options) &&
            isBackendAvailable(Backend::PlaMatrixVulkan, options.plaMatrixDevice))
        {
            const bool regular_scale = stats.cameraCount >= std::max(1, options.minPlaMatrixVulkanCameras) &&
                                       stats.observationCount >= std::max(1, options.minPlaMatrixVulkanObservations);
            return {Backend::PlaMatrixVulkan,
                    regular_scale ? "large_joint_problem_uses_plamatrix_vulkan"
                                  : "dense_joint_problem_uses_plamatrix_vulkan"};
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

    BackendDecision Solver::decideBackendForProblem(const Problem& problem, const SolveOptions& options)
    {
        return decideBackendForProblem(problem, makeCompatibilityOptions(options));
    }

    Backend Solver::selectBackendForProblem(const Problem& problem, const Options& options)
    {
        return decideBackendForProblem(problem, options).backend;
    }

    Backend Solver::selectBackendForProblem(const Problem& problem, const SolveOptions& options)
    {
        return decideBackendForProblem(problem, options).backend;
    }

} // namespace plabundle
