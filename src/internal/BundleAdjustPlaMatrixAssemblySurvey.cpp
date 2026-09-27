#include "BundleAdjustPlaMatrixAssemblyInternal.h"

#include "BundleAdjustPlaMatrixConstraints.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace plabundle::internal::plamatrix_ba::assembly_detail
{
    namespace
    {

        bool referenceTransitionParameter(std::size_t parameter)
        {
            return parameter != static_cast<std::size_t>(BAIntrinsicParameter::FocalAspectRatio);
        }

        void addCameraPoseResidual(plamatrix::internal::BlockNormalEquations<double>* equations,
                                   const CameraPosePrimaryTerms& terms,
                                   const ConstraintLinearization& linearization)
        {
            if (!equations || terms.count == 0)
            {
                return;
            }
            const std::size_t term_count = std::min(terms.count, terms.blocks.size());
            std::array<const double*, 2> jacobians{};
            for (std::size_t index = 0; index < term_count; ++index)
            {
                jacobians[index] = terms.jacobians[index].data();
            }
            equations->addPrimaryResidualBlocks(terms.blocks.data(),
                                                jacobians.data(),
                                                term_count,
                                                linearization.residual.data(),
                                                linearization.residualSize,
                                                linearization.normalWeight);
        }

        double assemblePrimaryPriors(const BAOptions& options,
                                     const ActiveProblem& active,
                                     const OptimizationState& state,
                                     int iteration,
                                     plamatrix::internal::BlockNormalEquations<double>* equations)
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
                    if (group.usesReferenceTransitionPrior && referenceTransitionParameter(parameter))
                    {
                        const double weight = group.transitionWeight[parameter];
                        if (parameter == static_cast<std::size_t>(BAIntrinsicParameter::FocalLength))
                        {
                            residual[parameter] = weight * (group.parameters[parameter] - group.focalReference);
                            jacobian[parameter * kBAIntrinsicParameterCount + parameter] = weight;
                        }
                        else
                        {
                            residual[parameter] = weight * (group.parameters[parameter] - group.prior[parameter]);
                            jacobian[parameter * kBAIntrinsicParameterCount + parameter] = weight;
                        }
                    }
                    else
                    {
                        residual[parameter] =
                            group.inverseSigma[parameter] * (group.parameters[parameter] - group.prior[parameter]);
                        jacobian[parameter * kBAIntrinsicParameterCount + parameter] = group.inverseSigma[parameter];
                    }
                    cost += 0.5 * residual[parameter] * residual[parameter];
                }
                if (equations)
                {
                    equations->addPrimaryResidualBlock(active.cameraBlockCount + static_cast<int>(group_index),
                                                       jacobian.data(),
                                                       residual.data(),
                                                       static_cast<int>(kBAIntrinsicParameterCount));
                }
            }
            for (std::size_t camera_index = 0; camera_index < state.cameras.size(); ++camera_index)
            {
                ConstraintLinearization linearization;
                if (camera_index < options.cameraPosePriors.size() &&
                    linearizePosePrior(
                        state.cameras[camera_index], options.cameraPosePriors[camera_index], options, &linearization))
                {
                    const CameraPosePrimaryTerms terms = cameraPosePrimaryTerms(options,
                                                                                active,
                                                                                state,
                                                                                camera_index,
                                                                                linearization.primaryJacobian.data(),
                                                                                linearization.residualSize,
                                                                                kPrimaryBlockSize);
                    if (terms.count > 0)
                    {
                        cost += linearization.robustCost;
                        addCameraPoseResidual(equations, terms, linearization);
                    }
                }
                if (linearizeCameraPlane(state.cameras[camera_index], camera_index, options, &linearization))
                {
                    const CameraPosePrimaryTerms terms = cameraPosePrimaryTerms(options,
                                                                                active,
                                                                                state,
                                                                                camera_index,
                                                                                linearization.primaryJacobian.data(),
                                                                                linearization.residualSize,
                                                                                kPrimaryBlockSize);
                    if (terms.count > 0)
                    {
                        cost += linearization.robustCost;
                        addCameraPoseResidual(equations, terms, linearization);
                    }
                }
            }
            return cost;
        }

        double assembleScaleBars(const BAOptions& options,
                                 const ActiveProblem& active,
                                 const OptimizationState& state,
                                 plamatrix::internal::BlockNormalEquations<double>* equations)
        {
            if (!options.enableScaleBarConstraints)
            {
                return 0.0;
            }
            double cost = 0.0;
            for (const auto& scale : options.scaleBarConstraints)
            {
                if (scale.trackIndexA < 0 || scale.trackIndexB < 0 ||
                    scale.trackIndexA >= static_cast<int>(state.points.size()) ||
                    scale.trackIndexB >= static_cast<int>(state.points.size()) ||
                    !active.activeTrack[static_cast<std::size_t>(scale.trackIndexA)] ||
                    !active.activeTrack[static_cast<std::size_t>(scale.trackIndexB)])
                {
                    continue;
                }
                ConstraintLinearization linearization;
                if (!linearizeScaleBar(scale,
                                       state.points[static_cast<std::size_t>(scale.trackIndexA)],
                                       state.points[static_cast<std::size_t>(scale.trackIndexB)],
                                       options,
                                       &linearization))
                {
                    continue;
                }
                cost += linearization.robustCost;
                if (!equations)
                {
                    continue;
                }
                std::vector<plamatrix::Index> blocks;
                std::vector<const double*> jacobians;
                const int block_a = active.trackPrimaryBlock[static_cast<std::size_t>(scale.trackIndexA)];
                const int block_b = active.trackPrimaryBlock[static_cast<std::size_t>(scale.trackIndexB)];
                if (block_a >= 0)
                {
                    blocks.push_back(block_a);
                    jacobians.push_back(linearization.primaryJacobian.data());
                }
                if (block_b >= 0)
                {
                    blocks.push_back(block_b);
                    jacobians.push_back(linearization.secondaryPrimaryJacobian.data());
                }
                if (!blocks.empty())
                {
                    equations->addPrimaryResidualBlocks(
                        blocks, jacobians, linearization.residual.data(), 1, linearization.normalWeight);
                }
            }
            return cost;
        }

        double assembleLaserRanges(const std::vector<CameraState>& input_cameras,
                                   const BAOptions& options,
                                   const ActiveProblem& active,
                                   const OptimizationState& state,
                                   int iteration,
                                   plamatrix::internal::BlockNormalEquations<double>* equations)
        {
            if (!options.enableLaserRangeConstraints)
            {
                return 0.0;
            }
            double cost = 0.0;
            for (std::size_t shot_index = 0; shot_index < options.laserRangeConstraints.size(); ++shot_index)
            {
                const auto& shot = options.laserRangeConstraints[shot_index];
                const auto& point = state.laserPoints[shot_index];
                ConstraintLinearization constraint;
                if (linearizeLaserRange(
                        state.cameras[static_cast<std::size_t>(shot.cameraIndex)], shot, point, options, &constraint))
                {
                    cost += constraint.robustCost;
                    const int point_block = active.laserBlock[shot_index];
                    const CameraPosePrimaryTerms terms =
                        cameraPosePrimaryTerms(options,
                                               active,
                                               state,
                                               static_cast<std::size_t>(shot.cameraIndex),
                                               constraint.primaryJacobian.data(),
                                               constraint.residualSize,
                                               kPrimaryBlockSize);
                    if (equations && terms.count > 0 && point_block >= 0)
                    {
                        const std::size_t term_count = std::min(terms.count, terms.blocks.size());
                        std::array<const double*, 2> jacobians{};
                        for (std::size_t index = 0; index < term_count; ++index)
                        {
                            jacobians[index] = terms.jacobians[index].data();
                        }
                        equations->addResidualBlocks(terms.blocks.data(),
                                                     jacobians.data(),
                                                     term_count,
                                                     point_block,
                                                     constraint.pointJacobian.data(),
                                                     constraint.residual.data(),
                                                     1,
                                                     constraint.normalWeight);
                    }
                    else if (terms.count > 0)
                    {
                        addCameraPoseResidual(equations, terms, constraint);
                    }
                    else
                    {
                        addPointResidual(equations,
                                         -1,
                                         point_block,
                                         constraint.pointJacobian.data(),
                                         constraint.residual.data(),
                                         1,
                                         constraint.normalWeight);
                    }
                }
                if (linearizeLaserPointPrior(shot, point, &constraint))
                {
                    cost += constraint.robustCost;
                    addPointResidual(equations,
                                     -1,
                                     active.laserBlock[shot_index],
                                     constraint.pointJacobian.data(),
                                     constraint.residual.data(),
                                     constraint.residualSize,
                                     1.0);
                }
                for (const auto& observation : shot.measuredImageObservations)
                {
                    const std::size_t camera_index = static_cast<std::size_t>(observation.cameraIndex);
                    ObservationLinearization linearization;
                    if (!linearizeImageObservation(input_cameras,
                                                   options,
                                                   active,
                                                   state,
                                                   camera_index,
                                                   point,
                                                   observation,
                                                   iteration,
                                                   &linearization))
                    {
                        throw InvalidProjectionError("PlaMatrix BA 激光测距像点线性化失败");
                    }
                    cost += linearization.robustCost;
                    addObservation(equations,
                                   options,
                                   active,
                                   state,
                                   camera_index,
                                   -1,
                                   active.laserBlock[shot_index],
                                   linearization);
                }
            }
            return cost;
        }

    } // namespace

    double assembleSurveyResiduals(const std::vector<CameraState>& input_cameras,
                                   const BAOptions& options,
                                   const ActiveProblem& active,
                                   const OptimizationState& state,
                                   int iteration,
                                   plamatrix::internal::BlockNormalEquations<double>* equations)
    {
        return assemblePrimaryPriors(options, active, state, iteration, equations) +
               assembleScaleBars(options, active, state, equations) +
               assembleLaserRanges(input_cameras, options, active, state, iteration, equations);
    }

} // namespace plabundle::internal::plamatrix_ba::assembly_detail
