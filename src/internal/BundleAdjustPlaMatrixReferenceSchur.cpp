#include "BundleAdjustPlaMatrixReferenceSchur.h"

#include <plamatrix/dense/small_inverse.h>

#include "BundleAdjustPlaMatrixAssemblyInternal.h"
#include "BundleAdjustPlaMatrixConstraints.h"
#include "BundleAdjustValidation.h"
#include "OpenMpCompat.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace plabundle::internal::plamatrix_ba
{
    void ReferenceSchurPointWorkspace::clear() noexcept
    {
        hessian.fill(0.0);
        rhs.fill(0.0);
        for (const plamatrix::Index block : primaryBlocks)
        {
            if (block >= 0 && static_cast<std::size_t>(block) < primaryBlockLookup.size())
            {
                primaryBlockLookup[static_cast<std::size_t>(block)] = -1;
            }
        }
        primaryBlocks.clear();
        crossBlocks.clear();
    }

    ReferenceSchurCrossBlock& ReferenceSchurPointWorkspace::crossBlock(plamatrix::Index block)
    {
        if (block >= 0 && static_cast<std::size_t>(block) < primaryBlockLookup.size())
        {
            const int index = primaryBlockLookup[static_cast<std::size_t>(block)];
            if (index >= 0)
            {
                return crossBlocks[static_cast<std::size_t>(index)];
            }
            primaryBlocks.push_back(block);
            crossBlocks.emplace_back();
            primaryBlockLookup[static_cast<std::size_t>(block)] = static_cast<int>(crossBlocks.size() - 1);
            return crossBlocks.back();
        }
        const auto found = std::find(primaryBlocks.begin(), primaryBlocks.end(), block);
        if (found != primaryBlocks.end())
        {
            return crossBlocks[static_cast<std::size_t>(found - primaryBlocks.begin())];
        }
        primaryBlocks.push_back(block);
        crossBlocks.emplace_back();
        return crossBlocks.back();
    }

    namespace
    {
        using PointNormalBlock = ReferenceSchurPointWorkspace;

        bool invertPointHessian(const std::array<double, 9>& matrix, std::array<double, 9>* inverse)
        {
            return plamatrix::tryInverse3x3RowMajor(matrix, inverse, 1.0e-15);
        }

        struct ReducedAccumulator
        {
            struct HessianBlockView
            {
                double* values = nullptr;
                bool transposed = false;
                bool atomic = false;

                void add(int row, int column, double value) const
                {
                    const int stored_row = transposed ? column : row;
                    const int stored_column = transposed ? row : column;
                    double& entry = values[static_cast<std::size_t>(stored_row * kPrimaryBlockSize + stored_column)];
                    if (atomic)
                    {
                        std::atomic_ref<double>(entry).fetch_add(value, std::memory_order_relaxed);
                    }
                    else
                    {
                        entry += value;
                    }
                }
            };

            std::vector<double>* diagonal = nullptr;
            std::vector<double>* directRhs = nullptr;
            std::vector<double>* schurRhs = nullptr;
            std::vector<double>* offDiagonal = nullptr;
            const std::vector<int>* offDiagonalLookup = nullptr;
            int primaryBlockCount = 0;
            bool ownsOffDiagonalSlot = false;

            void addGradientEntry(plamatrix::Index block, int parameter, double value, bool direct)
            {
                const std::size_t index = static_cast<std::size_t>(block * kPrimaryBlockSize + parameter);
                if (direct)
                {
                    (*directRhs)[index] -= value;
                }
                else
                {
                    (*schurRhs)[index] -= value;
                }
            }

            HessianBlockView hessianBlock(plamatrix::Index row_block, plamatrix::Index column_block) const
            {
                if (row_block == column_block)
                {
                    return {diagonal->data() +
                                static_cast<std::size_t>(row_block * kPrimaryBlockSize * kPrimaryBlockSize),
                            false, false};
                }

                const bool ordered = row_block < column_block;
                const plamatrix::Index stored_row = ordered ? row_block : column_block;
                const plamatrix::Index stored_column = ordered ? column_block : row_block;
                const std::size_t lookup_index =
                    static_cast<std::size_t>(stored_row) * static_cast<std::size_t>(primaryBlockCount) +
                    static_cast<std::size_t>(stored_column);
                const int slot = (*offDiagonalLookup)[lookup_index];
                if (slot < 0)
                {
                    throw std::logic_error("参考在线 Schur 缺少预计算的非对角块");
                }
                return {offDiagonal->data() + static_cast<std::size_t>(slot * kPrimaryBlockSize * kPrimaryBlockSize),
                        !ordered, !ownsOffDiagonalSlot};
            }
        };

        void accumulateObservationPrimaryTerms(const assembly_detail::ObservationPrimaryTerms& terms,
                                               const ObservationLinearization& linearization,
                                               ReducedAccumulator* accumulator)
        {
            std::array<std::array<int, kPrimaryBlockSize>, 3> active_parameters{};
            std::array<int, 3> active_counts{};
            for (std::size_t term = 0; term < terms.count; ++term)
            {
                for (int parameter = 0; parameter < kPrimaryBlockSize; ++parameter)
                {
                    const double jacobian_x = terms.jacobians[term][static_cast<std::size_t>(parameter)];
                    const double jacobian_y =
                        terms.jacobians[term][static_cast<std::size_t>(kPrimaryBlockSize + parameter)];
                    if (jacobian_x == 0.0 && jacobian_y == 0.0)
                    {
                        continue;
                    }
                    active_parameters[term][static_cast<std::size_t>(active_counts[term]++)] = parameter;
                    double gradient = 0.0;
                    for (int row = 0; row < 2; ++row)
                    {
                        gradient +=
                            linearization.normalWeight *
                            terms.jacobians[term][static_cast<std::size_t>(row * kPrimaryBlockSize + parameter)] *
                            linearization.residual[static_cast<std::size_t>(row)];
                    }
                    accumulator->addGradientEntry(terms.blocks[term], parameter, gradient, true);
                }
            }
            for (std::size_t left = 0; left < terms.count; ++left)
            {
                for (std::size_t right = left; right < terms.count; ++right)
                {
                    const auto hessian_block = accumulator->hessianBlock(terms.blocks[left], terms.blocks[right]);
                    for (int left_index = 0; left_index < active_counts[left]; ++left_index)
                    {
                        const int row = active_parameters[left][static_cast<std::size_t>(left_index)];
                        const double left_x = terms.jacobians[left][static_cast<std::size_t>(row)];
                        const double left_y = terms.jacobians[left][static_cast<std::size_t>(kPrimaryBlockSize + row)];
                        for (int right_index = 0; right_index < active_counts[right]; ++right_index)
                        {
                            const int column = active_parameters[right][static_cast<std::size_t>(right_index)];
                            const double right_x = terms.jacobians[right][static_cast<std::size_t>(column)];
                            const double right_y =
                                terms.jacobians[right][static_cast<std::size_t>(kPrimaryBlockSize + column)];
                            hessian_block.add(row,
                                              column,
                                              linearization.normalWeight *
                                                  (left_x * right_x + left_y * right_y));
                        }
                    }
                }
            }
        }

        void accumulatePointTerms(const assembly_detail::ObservationPrimaryTerms& terms,
                                  const ObservationLinearization& linearization,
                                  PointNormalBlock* point)
        {
            for (int column = 0; column < kEliminatedBlockSize; ++column)
            {
                for (int row = 0; row < 2; ++row)
                {
                    const double point_value =
                        linearization.pointJacobian[static_cast<std::size_t>(row * kEliminatedBlockSize + column)];
                    point->rhs[static_cast<std::size_t>(column)] -=
                        linearization.normalWeight * point_value *
                        linearization.residual[static_cast<std::size_t>(row)];
                    for (int other = 0; other < kEliminatedBlockSize; ++other)
                    {
                        point->hessian[static_cast<std::size_t>(column * kEliminatedBlockSize + other)] +=
                            linearization.normalWeight * point_value *
                            linearization.pointJacobian[static_cast<std::size_t>(row * kEliminatedBlockSize + other)];
                    }
                }
            }
            for (std::size_t term = 0; term < terms.count; ++term)
            {
                auto& cross = point->crossBlock(terms.blocks[term]);
                for (int primary = 0; primary < kPrimaryBlockSize; ++primary)
                {
                    const double jacobian_x = terms.jacobians[term][static_cast<std::size_t>(primary)];
                    const double jacobian_y =
                        terms.jacobians[term][static_cast<std::size_t>(kPrimaryBlockSize + primary)];
                    if (jacobian_x == 0.0 && jacobian_y == 0.0)
                    {
                        continue;
                    }
                    for (int eliminated = 0; eliminated < kEliminatedBlockSize; ++eliminated)
                    {
                        cross[static_cast<std::size_t>(primary * kEliminatedBlockSize + eliminated)] +=
                            linearization.normalWeight *
                            (jacobian_x * linearization.pointJacobian[static_cast<std::size_t>(eliminated)] +
                             jacobian_y *
                                 linearization
                                     .pointJacobian[static_cast<std::size_t>(kEliminatedBlockSize + eliminated)]);
                    }
                }
            }
        }

        void accumulatePointOnlyConstraint(const ConstraintLinearization& linearization, PointNormalBlock* point)
        {
            for (int column = 0; column < kEliminatedBlockSize; ++column)
            {
                for (int row = 0; row < linearization.residualSize; ++row)
                {
                    const double point_value =
                        linearization.pointJacobian[static_cast<std::size_t>(row * kEliminatedBlockSize + column)];
                    point->rhs[static_cast<std::size_t>(column)] -=
                        linearization.normalWeight * point_value *
                        linearization.residual[static_cast<std::size_t>(row)];
                    for (int other = 0; other < kEliminatedBlockSize; ++other)
                    {
                        point->hessian[static_cast<std::size_t>(column * kEliminatedBlockSize + other)] +=
                            linearization.normalWeight * point_value *
                            linearization.pointJacobian[static_cast<std::size_t>(row * kEliminatedBlockSize + other)];
                    }
                }
            }
        }

        bool linearizeTrack(const std::vector<CameraState>& input_cameras,
                            const std::vector<BATrack>& tracks,
                            const BAOptions& options,
                            const ActiveProblem& active,
                            const OptimizationState& state,
                            int iteration,
                            std::size_t track_index,
                            const EffectiveCameraCache* effective_camera_cache,
                            ReducedAccumulator* accumulator,
                            PointNormalBlock* point,
                            double* cost,
                            double* projection_seconds,
                            double* terms_accumulate_seconds,
                            std::size_t* observation_count)
        {
            point->clear();
            const bool eliminate_point = active.trackBlock[track_index] >= 0;
            for (const auto& observation : tracks[track_index].observations)
            {
                if (!observationIsUsable(observation, state.cameras.size()))
                {
                    continue;
                }
                const std::size_t camera_index = static_cast<std::size_t>(observation.cameraIndex);
                ObservationLinearization linearization;
                const auto projection_start = projection_seconds ? std::chrono::steady_clock::now()
                                                                  : std::chrono::steady_clock::time_point{};
                const bool valid_projection = assembly_detail::linearizeImageObservation(input_cameras,
                                                                                           options,
                                                                                           active,
                                                                                           state,
                                                                                           camera_index,
                                                                                           state.points[track_index],
                                                                                           observation,
                                                                                           iteration,
                                                                                           &linearization,
                                                                                           effective_camera_cache);
                if (projection_seconds)
                {
                    *projection_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - projection_start).count();
                }
                if (!valid_projection)
                {
                    throw InvalidProjectionError("参考在线 Schur 线性化产生非法投影");
                }
                *cost += linearization.robustCost;
                const auto terms_accumulate_start = terms_accumulate_seconds ? std::chrono::steady_clock::now()
                                                                              : std::chrono::steady_clock::time_point{};
                const auto terms =
                    assembly_detail::observationPrimaryTerms(options, active, state, camera_index, linearization);
                if (accumulator)
                {
                    accumulateObservationPrimaryTerms(terms, linearization, accumulator);
                }
                if (eliminate_point)
                {
                    accumulatePointTerms(terms, linearization, point);
                }
                if (terms_accumulate_seconds)
                {
                    *terms_accumulate_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - terms_accumulate_start).count();
                }
                if (observation_count)
                {
                    ++(*observation_count);
                }
            }
            if (options.enableControlPointConstraints)
            {
                ConstraintLinearization linearization;
                for (const BAControlPointConstraint& constraint : tracks[track_index].controlPointConstraints)
                {
                    if (!linearizeControlPoint(constraint, state.points[track_index], options, &linearization))
                    {
                        continue;
                    }
                    *cost += linearization.robustCost;
                    if (eliminate_point)
                    {
                        accumulatePointOnlyConstraint(linearization, point);
                    }
                }
            }
            return eliminate_point;
        }

        bool eliminatePoint(double damping,
                            PointNormalBlock* point,
                            ReducedAccumulator* accumulator,
                            ReferenceSchurBackSubstitutionData* back_substitution,
                            double* inverse_seconds,
                            double* reduction_seconds,
                            double* schur_accumulate_seconds,
                            double* back_substitution_copy_seconds)
        {
            const auto inverse_start = inverse_seconds ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
            std::array<double, 9> damped_hessian = point->hessian;
            for (int diagonal = 0; diagonal < kEliminatedBlockSize; ++diagonal)
            {
                damped_hessian[static_cast<std::size_t>(diagonal * kEliminatedBlockSize + diagonal)] *= 1.0 + damping;
            }
            std::array<double, 9> inverse{};
            if (!invertPointHessian(damped_hessian, &inverse))
            {
                if (back_substitution)
                {
                    back_substitution->valid = false;
                }
                return false;
            }
            if (inverse_seconds)
            {
                *inverse_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - inverse_start).count();
            }

            const auto reduction_start = reduction_seconds ? std::chrono::steady_clock::now()
                                                           : std::chrono::steady_clock::time_point{};
            point->reducedCrossBlocks.resize(point->crossBlocks.size());
            point->activeRows.resize(point->crossBlocks.size());
            for (std::size_t block = 0; block < point->crossBlocks.size(); ++block)
            {
                point->reducedCrossBlocks[block].fill(0.0);
                point->activeRows[block].fill(0);
                for (int row = 0; row < kPrimaryBlockSize; ++row)
                {
                    const std::size_t row_offset = static_cast<std::size_t>(row * kEliminatedBlockSize);
                    point->activeRows[block][static_cast<std::size_t>(row)] =
                        point->crossBlocks[block][row_offset] != 0.0 ||
                        point->crossBlocks[block][row_offset + 1] != 0.0 ||
                        point->crossBlocks[block][row_offset + 2] != 0.0;
                    if (!point->activeRows[block][static_cast<std::size_t>(row)])
                    {
                        continue;
                    }
                    for (int column = 0; column < kEliminatedBlockSize; ++column)
                    {
                        for (int inner = 0; inner < kEliminatedBlockSize; ++inner)
                        {
                            point->reducedCrossBlocks[block]
                                                     [static_cast<std::size_t>(row * kEliminatedBlockSize + column)] +=
                                point
                                    ->crossBlocks[block][static_cast<std::size_t>(row * kEliminatedBlockSize + inner)] *
                                inverse[static_cast<std::size_t>(inner * kEliminatedBlockSize + column)];
                        }
                    }
                }
                for (int row = 0; row < kPrimaryBlockSize; ++row)
                {
                    if (!point->activeRows[block][static_cast<std::size_t>(row)])
                    {
                        continue;
                    }
                    double gradient = 0.0;
                    for (int column = 0; column < kEliminatedBlockSize; ++column)
                    {
                        gradient +=
                            point->reducedCrossBlocks[block]
                                                     [static_cast<std::size_t>(row * kEliminatedBlockSize + column)] *
                            point->rhs[static_cast<std::size_t>(column)];
                    }
                    accumulator->addGradientEntry(point->primaryBlocks[block], row, gradient, false);
                }
            }
            if (reduction_seconds)
            {
                *reduction_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - reduction_start).count();
            }

            const auto schur_accumulate_start = schur_accumulate_seconds ? std::chrono::steady_clock::now()
                                                                          : std::chrono::steady_clock::time_point{};
            for (std::size_t left = 0; left < point->crossBlocks.size(); ++left)
            {
                for (std::size_t right = left; right < point->crossBlocks.size(); ++right)
                {
                    const auto hessian_block =
                        accumulator->hessianBlock(point->primaryBlocks[left], point->primaryBlocks[right]);
                    for (int row = 0; row < kPrimaryBlockSize; ++row)
                    {
                        if (!point->activeRows[left][static_cast<std::size_t>(row)])
                        {
                            continue;
                        }
                        for (int column = 0; column < kPrimaryBlockSize; ++column)
                        {
                            if (!point->activeRows[right][static_cast<std::size_t>(column)])
                            {
                                continue;
                            }
                            double hessian = 0.0;
                            for (int inner = 0; inner < kEliminatedBlockSize; ++inner)
                            {
                                hessian -=
                                    point->reducedCrossBlocks[left][static_cast<std::size_t>(
                                        row * kEliminatedBlockSize + inner)] *
                                    point->crossBlocks[right]
                                                      [static_cast<std::size_t>(column * kEliminatedBlockSize + inner)];
                            }
                            hessian_block.add(row, column, hessian);
                        }
                    }
                }
            }
            if (schur_accumulate_seconds)
            {
                *schur_accumulate_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - schur_accumulate_start).count();
            }
            if (back_substitution)
            {
                const auto copy_start = back_substitution_copy_seconds ? std::chrono::steady_clock::now()
                                                                        : std::chrono::steady_clock::time_point{};
                back_substitution->inverseHessian = inverse;
                back_substitution->rhs = point->rhs;
                back_substitution->primaryBlocks.assign(point->primaryBlocks.begin(), point->primaryBlocks.end());
                back_substitution->crossBlocks.assign(point->crossBlocks.begin(), point->crossBlocks.end());
                back_substitution->valid = true;
                if (back_substitution_copy_seconds)
                {
                    *back_substitution_copy_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - copy_start).count();
                }
            }
            return true;
        }

        double assembleReferencePriors(const BAOptions& options,
                                       const ActiveProblem& active,
                                       const OptimizationState& state,
                                       int iteration,
                                       plamatrix::internal::BlockNormalEquations<double>* equations,
                                       std::vector<double>* direct_rhs)
        {
            double cost = 0.0;
            for (std::size_t group_index = 0; group_index < state.intrinsicGroups.size(); ++group_index)
            {
                const auto& group = state.intrinsicGroups[group_index];
                const auto active_parameters = activeIntrinsicParameters(options, group.enabled, iteration);
                std::array<double, kBAIntrinsicParameterCount> residual{};
                std::array<double, kBAIntrinsicParameterCount * kBAIntrinsicParameterCount> jacobian{};
                for (std::size_t parameter = 0; parameter < kBAIntrinsicParameterCount; ++parameter)
                {
                    if (!active_parameters[parameter])
                    {
                        continue;
                    }
                    const bool transition_parameter =
                        parameter != static_cast<std::size_t>(BAIntrinsicParameter::FocalAspectRatio);
                    const bool use_transition = group.usesReferenceTransitionPrior && transition_parameter;
                    const double reference =
                        use_transition && parameter == static_cast<std::size_t>(BAIntrinsicParameter::FocalLength)
                            ? group.focalReference
                            : group.prior[parameter];
                    const double weight =
                        use_transition ? group.transitionWeight[parameter] : group.inverseSigma[parameter];
                    residual[parameter] = weight * (group.parameters[parameter] - reference);
                    jacobian[parameter * kBAIntrinsicParameterCount + parameter] = weight;
                    cost += 0.5 * residual[parameter] * residual[parameter];
                }
                const int block = active.cameraBlockCount + static_cast<int>(group_index);
                equations->addPrimaryResidualBlock(
                    block, jacobian.data(), residual.data(), static_cast<int>(kBAIntrinsicParameterCount));
                const std::size_t offset = static_cast<std::size_t>(block * kPrimaryBlockSize);
                for (std::size_t parameter = 0; parameter < kBAIntrinsicParameterCount; ++parameter)
                {
                    (*direct_rhs)[offset + parameter] -=
                        jacobian[parameter * kBAIntrinsicParameterCount + parameter] * residual[parameter];
                }
            }
            return cost;
        }

        void updateTrackBoundaries(const std::vector<BATrack>& tracks,
                                   const ActiveProblem& active,
                                   int thread_count,
                                   ReferenceSchurWorkspace* workspace)
        {
            if (workspace->partitionThreadCount == thread_count &&
                workspace->trackBoundaries.size() == static_cast<std::size_t>(thread_count + 1))
            {
                return;
            }
            std::vector<std::size_t> prefix_cost(tracks.size() + 1, 0);
            for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
            {
                const std::size_t work = active.activeTrack[track_index]
                                             ? std::max<std::size_t>(1, tracks[track_index].observations.size())
                                             : 1;
                prefix_cost[track_index + 1] = prefix_cost[track_index] + work;
            }
            workspace->trackBoundaries.assign(static_cast<std::size_t>(thread_count + 1), tracks.size());
            workspace->trackBoundaries[0] = 0;
            for (int thread = 1; thread < thread_count; ++thread)
            {
                const std::size_t target =
                    prefix_cost.back() * static_cast<std::size_t>(thread) / static_cast<std::size_t>(thread_count);
                workspace->trackBoundaries[static_cast<std::size_t>(thread)] = static_cast<std::size_t>(
                    std::lower_bound(prefix_cost.begin(), prefix_cost.end(), target) - prefix_cost.begin());
            }
            workspace->partitionThreadCount = thread_count;
        }

        void prepareOffDiagonalTopology(const std::vector<BATrack>& tracks,
                                        const ActiveProblem& active,
                                        ReferenceSchurWorkspace* workspace)
        {
            if (!workspace->offDiagonalLookup.empty())
            {
                return;
            }
            const int block_count = active.primaryBlockCount;
            const std::size_t lookup_size =
                static_cast<std::size_t>(block_count) * static_cast<std::size_t>(block_count);
            workspace->offDiagonalLookup.assign(lookup_size, -1);
            std::vector<char> connected(lookup_size, 0);
            const auto connect = [&](int left, int right)
            {
                if (left < 0 || right < 0 || left == right)
                {
                    return;
                }
                const int row = std::min(left, right);
                const int column = std::max(left, right);
                connected[static_cast<std::size_t>(row) * static_cast<std::size_t>(block_count) +
                          static_cast<std::size_t>(column)] = 1;
            };

            std::vector<int> track_blocks;
            for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
            {
                if (!active.activeTrack[track_index])
                {
                    continue;
                }
                track_blocks.clear();
                for (const BAObservation& observation : tracks[track_index].observations)
                {
                    if (!observationIsUsable(observation, active.cameraBlock.size()))
                    {
                        continue;
                    }
                    const std::size_t camera_index = static_cast<std::size_t>(observation.cameraIndex);
                    const std::array<int, 4> observation_blocks{{active.cameraBlock[camera_index],
                                                                 active.rigCaptureBlockByCamera[camera_index],
                                                                 active.rigSensorBlockByCamera[camera_index],
                                                                 active.intrinsicBlockByCamera[camera_index]}};
                    for (std::size_t left = 0; left < observation_blocks.size(); ++left)
                    {
                        for (std::size_t right = left + 1; right < observation_blocks.size(); ++right)
                        {
                            connect(observation_blocks[left], observation_blocks[right]);
                        }
                    }
                    if (active.trackBlock[track_index] >= 0)
                    {
                        for (const int block : observation_blocks)
                        {
                            if (block >= 0)
                            {
                                track_blocks.push_back(block);
                            }
                        }
                    }
                }
                if (active.trackBlock[track_index] < 0)
                {
                    continue;
                }
                std::sort(track_blocks.begin(), track_blocks.end());
                track_blocks.erase(std::unique(track_blocks.begin(), track_blocks.end()), track_blocks.end());
                for (std::size_t left = 0; left < track_blocks.size(); ++left)
                {
                    for (std::size_t right = left + 1; right < track_blocks.size(); ++right)
                    {
                        connect(track_blocks[left], track_blocks[right]);
                    }
                }
            }

            for (int row = 0; row < block_count; ++row)
            {
                for (int column = row + 1; column < block_count; ++column)
                {
                    const std::size_t lookup = static_cast<std::size_t>(row) * static_cast<std::size_t>(block_count) +
                                               static_cast<std::size_t>(column);
                    if (!connected[lookup])
                    {
                        continue;
                    }
                    workspace->offDiagonalLookup[lookup] = static_cast<int>(workspace->offDiagonalBlocks.size());
                    workspace->offDiagonalBlocks.emplace_back(row, column);
                }
            }
        }
    } // namespace

    ReferenceSchurWorkspace::ReferenceSchurWorkspace(const ActiveProblem& active)
        : reducedEquations(active.primaryBlockCount, 0, kPrimaryBlockSize, kEliminatedBlockSize)
    {
    }

    bool canUseReferenceOnlineSchur(const BAOptions& options, const ActiveProblem& active)
    {
        return options.useReferenceOnlineSchur && active.primaryBlockCount > 0 && active.promotedTrackBlockCount == 0 &&
               active.laserBlockCount == 0 && !options.enableLaserPlaneConstraints &&
               !options.enableLaserRangeConstraints && !options.enableScaleBarConstraints &&
               options.cameraPosePriors.empty() && !options.cameraPlaneConstraint.enabled;
    }

    ReferenceSchurBuildResult buildReferenceReducedNormalEquations(const std::vector<CameraState>& input_cameras,
                                                                   const std::vector<BATrack>& tracks,
                                                                   const BAOptions& options,
                                                                   const ActiveProblem& active,
                                                                   const OptimizationState& state,
                                                                   int iteration,
                                                                   double damping,
                                                                   ReferenceSchurWorkspace* workspace)
    {
        const bool trace_assembly = std::getenv("METALIGN_TRACE_PLABUNDLE_ASSEMBLY") != nullptr;
        const auto total_start = std::chrono::steady_clock::now();
        if (!workspace || !std::isfinite(damping) || damping < 0.0)
        {
            throw std::invalid_argument("参考在线 Schur 工作区或阻尼无效");
        }
        const char* assembly_thread_override = std::getenv("METALIGN_PLABUNDLE_ASSEMBLY_THREADS");
        int requested_threads = options.numThreads > 0 ? options.numThreads : openMpMaxThreads();
        if (assembly_thread_override)
        {
            const int override_threads = std::atoi(assembly_thread_override);
            if (override_threads > 0)
            {
                requested_threads = override_threads;
            }
        }
        const int thread_count = std::min<int>(std::max(1, requested_threads), static_cast<int>(tracks.size()));
        int effective_thread_count = thread_count;
        const char* serial_track_threshold = std::getenv("METALIGN_PLABUNDLE_REFERENCE_SERIAL_TRACK_THRESHOLD");
        if (serial_track_threshold)
        {
            const int threshold = std::atoi(serial_track_threshold);
            if (threshold > 0 && tracks.size() <= static_cast<std::size_t>(threshold))
            {
                effective_thread_count = 1;
            }
        }
        const auto setup_start = std::chrono::steady_clock::now();
        assembly_detail::prepareEffectiveCameraCache(
            input_cameras, options, active, state, iteration, &workspace->effectiveCameraCache);
        updateTrackBoundaries(tracks, active, effective_thread_count, workspace);
        prepareOffDiagonalTopology(tracks, active, workspace);
        workspace->partialDiagonals.resize(static_cast<std::size_t>(effective_thread_count));
        workspace->partialDirectRhs.resize(static_cast<std::size_t>(effective_thread_count));
        workspace->partialSchurRhs.resize(static_cast<std::size_t>(effective_thread_count));
        workspace->partialCosts.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
        workspace->partialPointSystemSingular.assign(static_cast<std::size_t>(effective_thread_count), 0);
        workspace->errors.assign(static_cast<std::size_t>(effective_thread_count), {});
        workspace->pointWorkspaces.resize(static_cast<std::size_t>(effective_thread_count));
        workspace->backSubstitutionData.resize(tracks.size());
        std::vector<double> partial_linearize_seconds;
        std::vector<double> partial_eliminate_seconds;
        std::vector<double> partial_projection_seconds;
        std::vector<double> partial_terms_accumulate_seconds;
        std::vector<std::size_t> partial_observation_counts;
        std::vector<double> partial_inverse_seconds;
        std::vector<double> partial_reduction_seconds;
        std::vector<double> partial_schur_accumulate_seconds;
        std::vector<double> partial_back_substitution_copy_seconds;
        if (trace_assembly)
        {
            partial_linearize_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_eliminate_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_projection_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_terms_accumulate_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_observation_counts.assign(static_cast<std::size_t>(effective_thread_count), 0);
            partial_inverse_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_reduction_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_schur_accumulate_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
            partial_back_substitution_copy_seconds.assign(static_cast<std::size_t>(effective_thread_count), 0.0);
        }
        const std::size_t primary_dimension = static_cast<std::size_t>(active.primaryBlockCount * kPrimaryBlockSize);
        const std::size_t diagonal_dimension =
            static_cast<std::size_t>(active.primaryBlockCount * kPrimaryBlockSize * kPrimaryBlockSize);
        for (int thread = 0; thread < effective_thread_count; ++thread)
        {
            const std::size_t index = static_cast<std::size_t>(thread);
            workspace->partialDiagonals[index].assign(diagonal_dimension, 0.0);
            workspace->partialDirectRhs[index].assign(primary_dimension, 0.0);
            workspace->partialSchurRhs[index].assign(primary_dimension, 0.0);
        }
        const std::size_t off_diagonal_dimension =
            workspace->offDiagonalBlocks.size() * static_cast<std::size_t>(kPrimaryBlockSize * kPrimaryBlockSize);
        // Give each worker an exclusive slot while that remains inexpensive.
        // Dense, very large camera graphs fall back to shared atomic slots so
        // the workspace cannot multiply without bound with the thread count.
        constexpr std::size_t kOffDiagonalSlotBudgetBytes = 128ULL * 1024ULL * 1024ULL;
        const std::size_t bytes_per_slot = off_diagonal_dimension * sizeof(double);
        const std::size_t slots_within_budget =
            bytes_per_slot == 0 ? static_cast<std::size_t>(effective_thread_count)
                                : std::max<std::size_t>(1, kOffDiagonalSlotBudgetBytes / bytes_per_slot);
        const int shared_slot_count = slots_within_budget >= static_cast<std::size_t>(effective_thread_count)
                                          ? effective_thread_count
                                          : static_cast<int>(slots_within_budget);
        const bool workers_own_off_diagonal_slots = shared_slot_count == effective_thread_count;
        workspace->sharedOffDiagonalSlots.resize(static_cast<std::size_t>(shared_slot_count));
        for (auto& slot : workspace->sharedOffDiagonalSlots)
        {
            slot.assign(off_diagonal_dimension, 0.0);
        }
        const double setup_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();

        const auto parallel_start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(effective_thread_count) schedule(static, 1)
        for (int thread = 0; thread < effective_thread_count; ++thread)
        {
            try
            {
                PointNormalBlock& point = workspace->pointWorkspaces[static_cast<std::size_t>(thread)];
                point.primaryBlocks.reserve(16);
                if (point.primaryBlockLookup.size() != static_cast<std::size_t>(active.primaryBlockCount))
                {
                    point.primaryBlockLookup.assign(static_cast<std::size_t>(active.primaryBlockCount), -1);
                }
                point.crossBlocks.reserve(16);
                point.reducedCrossBlocks.reserve(16);
                point.activeRows.reserve(16);
                const std::size_t thread_index = static_cast<std::size_t>(thread);
                ReducedAccumulator accumulator{
                    &workspace->partialDiagonals[thread_index],
                    &workspace->partialDirectRhs[thread_index],
                    &workspace->partialSchurRhs[thread_index],
                    &workspace->sharedOffDiagonalSlots[static_cast<std::size_t>(thread % shared_slot_count)],
                    &workspace->offDiagonalLookup,
                    active.primaryBlockCount,
                    workers_own_off_diagonal_slots,
                };
                double cost = 0.0;
                const std::size_t begin = workspace->trackBoundaries[thread_index];
                const std::size_t end = workspace->trackBoundaries[thread_index + 1];
                for (std::size_t track_index = begin; track_index < end; ++track_index)
                {
                    if (!active.activeTrack[track_index])
                    {
                        continue;
                    }
                    const auto linearize_start = std::chrono::steady_clock::now();
                    const bool linearized = linearizeTrack(input_cameras,
                                                           tracks,
                                                           options,
                                                           active,
                                                           state,
                                                           iteration,
                                                           track_index,
                                                           &workspace->effectiveCameraCache,
                                                           &accumulator,
                                                           &point,
                                                           &cost,
                                                           trace_assembly ? &partial_projection_seconds[thread_index]
                                                                          : nullptr,
                                                           trace_assembly ? &partial_terms_accumulate_seconds[thread_index]
                                                                          : nullptr,
                                                           trace_assembly ? &partial_observation_counts[thread_index]
                                                                          : nullptr);
                    if (trace_assembly)
                    {
                        partial_linearize_seconds[thread_index] += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - linearize_start).count();
                    }
                    if (linearized)
                    {
                        const auto eliminate_start = std::chrono::steady_clock::now();
                        if (!eliminatePoint(damping,
                                            &point,
                                            &accumulator,
                                            &workspace->backSubstitutionData[track_index],
                                            trace_assembly ? &partial_inverse_seconds[thread_index] : nullptr,
                                            trace_assembly ? &partial_reduction_seconds[thread_index] : nullptr,
                                            trace_assembly ? &partial_schur_accumulate_seconds[thread_index] : nullptr,
                                            trace_assembly ? &partial_back_substitution_copy_seconds[thread_index]
                                                           : nullptr))
                        {
                            workspace->partialPointSystemSingular[thread_index] = 1;
                        }
                        if (trace_assembly)
                        {
                            partial_eliminate_seconds[thread_index] += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - eliminate_start).count();
                        }
                    }
                }
                workspace->partialCosts[thread_index] = cost;
            }
            catch (...)
            {
                workspace->errors[static_cast<std::size_t>(thread)] = std::current_exception();
            }
        }
        const double parallel_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - parallel_start).count();

        const auto merge_start = std::chrono::steady_clock::now();
        workspace->reducedEquations.clearValues();
        workspace->directPrimaryRhs.assign(primary_dimension, 0.0);
        ReferenceSchurBuildResult result;
        for (int thread = 0; thread < effective_thread_count; ++thread)
        {
            const auto index = static_cast<std::size_t>(thread);
            if (workspace->errors[index])
            {
                std::rethrow_exception(workspace->errors[index]);
            }
            for (std::size_t parameter = 0; parameter < primary_dimension; ++parameter)
            {
                workspace->directPrimaryRhs[parameter] += workspace->partialDirectRhs[index][parameter];
            }
            result.objectiveCost += workspace->partialCosts[index];
            result.pointSystemSingular =
                result.pointSystemSingular || workspace->partialPointSystemSingular[index] != 0;
        }
        for (int block = 0; block < active.primaryBlockCount; ++block)
        {
            std::array<double, kPrimaryBlockSize> gradient{};
            std::array<double, kPrimaryBlockSize * kPrimaryBlockSize> diagonal{};
            const std::size_t gradient_offset = static_cast<std::size_t>(block * kPrimaryBlockSize);
            const std::size_t diagonal_offset = static_cast<std::size_t>(block * kPrimaryBlockSize * kPrimaryBlockSize);
            for (int thread = 0; thread < effective_thread_count; ++thread)
            {
                const auto& partial_diagonal = workspace->partialDiagonals[static_cast<std::size_t>(thread)];
                const auto& partial_schur_rhs = workspace->partialSchurRhs[static_cast<std::size_t>(thread)];
                for (int parameter = 0; parameter < kPrimaryBlockSize; ++parameter)
                {
                    gradient[static_cast<std::size_t>(parameter)] -=
                        partial_schur_rhs[gradient_offset + static_cast<std::size_t>(parameter)];
                }
                for (int parameter = 0; parameter < kPrimaryBlockSize * kPrimaryBlockSize; ++parameter)
                {
                    diagonal[static_cast<std::size_t>(parameter)] +=
                        partial_diagonal[diagonal_offset + static_cast<std::size_t>(parameter)];
                }
            }
            for (int parameter = 0; parameter < kPrimaryBlockSize; ++parameter)
            {
                gradient[static_cast<std::size_t>(parameter)] -=
                    workspace->directPrimaryRhs[gradient_offset + static_cast<std::size_t>(parameter)];
            }
            workspace->reducedEquations.addPrimaryGradientBlock(block, gradient.data());
            workspace->reducedEquations.addPrimaryHessianBlock(block, block, diagonal.data());
        }
        for (std::size_t block = 0; block < workspace->offDiagonalBlocks.size(); ++block)
        {
            std::array<double, kPrimaryBlockSize * kPrimaryBlockSize> hessian{};
            const std::size_t offset = block * static_cast<std::size_t>(kPrimaryBlockSize * kPrimaryBlockSize);
            for (const auto& slot : workspace->sharedOffDiagonalSlots)
            {
                for (int parameter = 0; parameter < kPrimaryBlockSize * kPrimaryBlockSize; ++parameter)
                {
                    hessian[static_cast<std::size_t>(parameter)] += slot[offset + static_cast<std::size_t>(parameter)];
                }
            }
            workspace->reducedEquations.addPrimaryHessianBlock(
                workspace->offDiagonalBlocks[block].first, workspace->offDiagonalBlocks[block].second, hessian.data());
        }
        result.objectiveCost += assembleReferencePriors(
            options, active, state, iteration, &workspace->reducedEquations, &workspace->directPrimaryRhs);
        workspace->reducedEquations.finalizePrimaryDiagonal(1.0 + damping, 1.0);
        if (trace_assembly)
        {
            const double merge_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - merge_start).count();
            const double total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - total_start).count();
            const std::size_t active_tracks = static_cast<std::size_t>(std::count(
                active.activeTrack.begin(), active.activeTrack.end(), true));
            const double linearize_seconds = std::accumulate(
                partial_linearize_seconds.begin(), partial_linearize_seconds.end(), 0.0);
            const double eliminate_seconds = std::accumulate(
                partial_eliminate_seconds.begin(), partial_eliminate_seconds.end(), 0.0);
            const double projection_seconds = std::accumulate(
                partial_projection_seconds.begin(), partial_projection_seconds.end(), 0.0);
            const double terms_accumulate_seconds = std::accumulate(
                partial_terms_accumulate_seconds.begin(), partial_terms_accumulate_seconds.end(), 0.0);
            const std::size_t observation_count = std::accumulate(
                partial_observation_counts.begin(), partial_observation_counts.end(), static_cast<std::size_t>(0));
            const double inverse_seconds = std::accumulate(
                partial_inverse_seconds.begin(), partial_inverse_seconds.end(), 0.0);
            const double reduction_seconds = std::accumulate(
                partial_reduction_seconds.begin(), partial_reduction_seconds.end(), 0.0);
            const double schur_accumulate_seconds = std::accumulate(
                partial_schur_accumulate_seconds.begin(), partial_schur_accumulate_seconds.end(), 0.0);
            const double back_substitution_copy_seconds = std::accumulate(
                partial_back_substitution_copy_seconds.begin(), partial_back_substitution_copy_seconds.end(), 0.0);
            std::cerr << "TRACE_PLABUNDLE_REFERENCE_ASSEMBLY tracks=" << tracks.size()
                      << " active_tracks=" << active_tracks
                      << " primary_blocks=" << active.primaryBlockCount
                      << " threads=" << effective_thread_count
                      << " requested_threads=" << thread_count
                      << " setup=" << setup_seconds
                      << " parallel=" << parallel_seconds
                      << " linearize=" << linearize_seconds
                      << " projection=" << projection_seconds
                      << " terms_accumulate=" << terms_accumulate_seconds
                      << " observations=" << observation_count
                      << " eliminate=" << eliminate_seconds
                      << " inverse=" << inverse_seconds
                      << " reduction=" << reduction_seconds
                      << " schur_accumulate=" << schur_accumulate_seconds
                      << " back_substitution_copy=" << back_substitution_copy_seconds
                      << " merge=" << merge_seconds
                      << " total=" << total_seconds
                      << " off_diagonal_blocks=" << workspace->offDiagonalBlocks.size()
                      << " shared_slots=" << workspace->sharedOffDiagonalSlots.size()
                      << '\n';
        }
        return result;
    }

    ReferenceSchurBackSubstitutionResult recoverReferencePointSteps(const std::vector<CameraState>& input_cameras,
                                                                    const std::vector<BATrack>& tracks,
                                                                    const BAOptions& options,
                                                                    const ActiveProblem& active,
                                                                    const OptimizationState& state,
                                                                    int iteration,
                                                                    double damping,
                                                                    const std::vector<double>& primary_step,
                                                                    const std::vector<double>& direct_primary_rhs,
                                                                    ReferenceSchurWorkspace* workspace,
                                                                    std::vector<double>* eliminated_step)
    {
        ReferenceSchurBackSubstitutionResult result;
        if (!workspace || !eliminated_step)
        {
            throw std::invalid_argument("参考在线 Schur 回代工作区或输出为空");
        }
        static_cast<void>(input_cameras);
        static_cast<void>(state);
        static_cast<void>(iteration);
        static_cast<void>(damping);
        eliminated_step->assign(static_cast<std::size_t>(active.trackBlockCount * kEliminatedBlockSize), 0.0);
        for (std::size_t parameter = 0; parameter < primary_step.size(); ++parameter)
        {
            result.directionalDecrease += primary_step[parameter] * direct_primary_rhs[parameter];
        }
        const int requested_threads = options.numThreads > 0 ? options.numThreads : openMpMaxThreads();
        const int thread_count = std::min<int>(std::max(1, requested_threads), static_cast<int>(tracks.size()));
        updateTrackBoundaries(tracks, active, thread_count, workspace);
        workspace->partialDirectionalDecrease.assign(static_cast<std::size_t>(thread_count), 0.0);
        workspace->partialBackSubstitutionSuccess.assign(static_cast<std::size_t>(thread_count), 1);
        workspace->errors.assign(static_cast<std::size_t>(thread_count), {});

#pragma omp parallel for num_threads(thread_count) schedule(static, 1)
        for (int thread = 0; thread < thread_count; ++thread)
        {
            try
            {
                const std::size_t begin = workspace->trackBoundaries[static_cast<std::size_t>(thread)];
                const std::size_t end = workspace->trackBoundaries[static_cast<std::size_t>(thread + 1)];
                for (std::size_t track_index = begin; track_index < end; ++track_index)
                {
                    const int point_block = active.trackBlock[track_index];
                    if (!active.activeTrack[track_index] || point_block < 0)
                    {
                        continue;
                    }
                    const ReferenceSchurBackSubstitutionData& point = workspace->backSubstitutionData[track_index];
                    if (!point.valid)
                    {
                        workspace->partialBackSubstitutionSuccess[static_cast<std::size_t>(thread)] = 0;
                        continue;
                    }
                    std::array<double, 3> conditioned_rhs = point.rhs;
                    for (std::size_t block = 0; block < point.primaryBlocks.size(); ++block)
                    {
                        const std::size_t primary_offset =
                            static_cast<std::size_t>(point.primaryBlocks[block] * kPrimaryBlockSize);
                        for (int eliminated = 0; eliminated < kEliminatedBlockSize; ++eliminated)
                        {
                            for (int primary = 0; primary < kPrimaryBlockSize; ++primary)
                            {
                                conditioned_rhs[static_cast<std::size_t>(eliminated)] -=
                                    point.crossBlocks[block][static_cast<std::size_t>(primary * kEliminatedBlockSize +
                                                                                      eliminated)] *
                                    primary_step[primary_offset + static_cast<std::size_t>(primary)];
                            }
                        }
                    }
                    const std::size_t output_offset = static_cast<std::size_t>(point_block * kEliminatedBlockSize);
                    for (int row = 0; row < kEliminatedBlockSize; ++row)
                    {
                        for (int column = 0; column < kEliminatedBlockSize; ++column)
                        {
                            (*eliminated_step)[output_offset + static_cast<std::size_t>(row)] +=
                                point.inverseHessian[static_cast<std::size_t>(row * kEliminatedBlockSize + column)] *
                                conditioned_rhs[static_cast<std::size_t>(column)];
                        }
                        workspace->partialDirectionalDecrease[static_cast<std::size_t>(thread)] +=
                            point.rhs[static_cast<std::size_t>(row)] *
                            (*eliminated_step)[output_offset + static_cast<std::size_t>(row)];
                    }
                }
            }
            catch (...)
            {
                workspace->errors[static_cast<std::size_t>(thread)] = std::current_exception();
            }
        }
        for (int thread = 0; thread < thread_count; ++thread)
        {
            const auto index = static_cast<std::size_t>(thread);
            if (workspace->errors[index])
            {
                std::rethrow_exception(workspace->errors[index]);
            }
            result.directionalDecrease += workspace->partialDirectionalDecrease[index];
            result.success = result.success && workspace->partialBackSubstitutionSuccess[index] != 0;
        }
        return result;
    }

} // namespace plabundle::internal::plamatrix_ba
