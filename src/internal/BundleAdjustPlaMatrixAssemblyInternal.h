#pragma once

#include "BundleAdjustPlaMatrixAssembly.h"
#include "BundleAdjustPlaMatrixProjection.h"

namespace plabundle::internal::plamatrix_ba::assembly_detail
{

    struct CameraPosePrimaryTerms
    {
        std::array<plamatrix::Index, 2> blocks{};
        std::array<std::array<double, 6 * kPrimaryBlockSize>, 2> jacobians{};
        std::size_t count = 0;
    };

    struct ObservationPrimaryTerms
    {
        std::array<plamatrix::Index, 3> blocks{};
        std::array<std::array<double, 2 * kPrimaryBlockSize>, 3> jacobians{};
        std::size_t count = 0;
    };

    CameraPosePrimaryTerms cameraPosePrimaryTerms(const BAOptions& options,
                                                  const ActiveProblem& active,
                                                  const OptimizationState& state,
                                                  std::size_t camera_index,
                                                  const double* camera_jacobian,
                                                  int residual_size,
                                                  int row_stride);

    ObservationPrimaryTerms observationPrimaryTerms(const BAOptions& options,
                                                    const ActiveProblem& active,
                                                    const OptimizationState& state,
                                                    std::size_t camera_index,
                                                    const ObservationLinearization& linearization);

    void addPointResidual(plamatrix::internal::BlockNormalEquations<double>* equations,
                          int primary_block,
                          int eliminated_block,
                          const double* point_jacobian,
                          const double* residual,
                          int residual_size,
                          double weight);

    void addObservation(plamatrix::internal::BlockNormalEquations<double>* equations,
                        const BAOptions& options,
                        const ActiveProblem& active,
                        const OptimizationState& state,
                        std::size_t camera_index,
                        int point_primary_block,
                        int point_eliminated_block,
                        const ObservationLinearization& linearization);

    bool linearizeImageObservation(const std::vector<CameraState>& input_cameras,
                                   const BAOptions& options,
                                   const ActiveProblem& active,
                                   const OptimizationState& state,
                                   std::size_t camera_index,
                                   const std::array<double, 3>& point,
                                   const BAObservation& observation,
                                   int iteration,
                                   ObservationLinearization* output);

    double assembleSurveyResiduals(const std::vector<CameraState>& input_cameras,
                                   const BAOptions& options,
                                   const ActiveProblem& active,
                                   const OptimizationState& state,
                                   int iteration,
                                   plamatrix::internal::BlockNormalEquations<double>* equations);

} // namespace plabundle::internal::plamatrix_ba::assembly_detail
