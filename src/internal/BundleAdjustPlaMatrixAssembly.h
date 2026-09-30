#pragma once

#include "BundleAdjustPlaMatrixProblem.h"

#include <plamatrix/internal/optimization/block_schur.h>

#include <exception>
#include <memory>
#include <stdexcept>

namespace plabundle::internal::plamatrix_ba
{

    using placamera::RigCapture;
    using placamera::RigSensor;
    using placamera::RigTopology;

    class InvalidProjectionError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    struct OptimizationState
    {
        std::vector<CameraState> cameras;
        RigTopology rig;
        std::vector<std::array<double, 3>> points;
        std::vector<std::array<double, 3>> laserPoints;
        std::vector<IntrinsicGroupState> intrinsicGroups;
    };

    struct EffectiveCameraCache
    {
        std::vector<CameraState> cameras;
        std::vector<BAIntrinsicParameterMask> activeParameters;
    };

    struct NormalEquationAssemblyWorkspace
    {
        explicit NormalEquationAssemblyWorkspace(const ActiveProblem& active);

        plamatrix::internal::BlockNormalEquations<double> equations;
        std::vector<std::unique_ptr<plamatrix::internal::BlockNormalEquations<double>>> partialEquations;
        std::vector<double> partialCosts;
        std::vector<std::exception_ptr> errors;
        std::vector<std::size_t> trackBoundaries;
        EffectiveCameraCache effectiveCameraCache;
        int partitionThreadCount = 0;
    };

    OptimizationState initializeState(const std::vector<CameraState>& cameras,
                                      const std::vector<BATrack>& tracks,
                                      const BAOptions& options,
                                      const ActiveProblem& active);

    void buildNormalEquations(const std::vector<CameraState>& input_cameras,
                              const std::vector<BATrack>& tracks,
                              const BAOptions& options,
                              const ActiveProblem& active,
                              const OptimizationState& state,
                              int iteration,
                              NormalEquationAssemblyWorkspace* workspace,
                              double* objective_cost = nullptr);

    double evaluateObjective(const std::vector<CameraState>& input_cameras,
                             const std::vector<BATrack>& tracks,
                             const BAOptions& options,
                             const ActiveProblem& active,
                             const OptimizationState& state,
                             int iteration,
                             NormalEquationAssemblyWorkspace* workspace = nullptr);

    double maximumStepNorm(const std::vector<double>& primary_step, const std::vector<double>& eliminated_step);

    BAIntrinsicParameterMask committedReferenceIntrinsicParameters(const BAOptions& options,
                                                                   const ActiveProblem& active,
                                                                   const OptimizationState& state,
                                                                   double final_cost);

    void applyStep(const ActiveProblem& active,
                   const std::vector<double>& primary_step,
                   const std::vector<double>& eliminated_step,
                   OptimizationState* state,
                   const BAOptions& options,
                   double step_scale = 1.0);

} // namespace plabundle::internal::plamatrix_ba
