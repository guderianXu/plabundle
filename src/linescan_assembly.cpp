#include "linescan_assembly.h"

#include <plamatrix/internal/optimization/robust_loss.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#ifdef PLABUNDLE_ENABLE_OPENMP
#include <omp.h>
#endif

namespace plabundle::linescan::detail::assembly
{
    namespace
    {

        using Vector3 = std::array<double, 3>;

        int maxThreads()
        {
#ifdef PLABUNDLE_ENABLE_OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }

        bool linearizeImageObservation(const Problem& problem,
                                       const ImageObservation& observation,
                                       const Options& options,
                                       double* residual,
                                       double* cameraJacobian,
                                       double* pointJacobian)
        {
            const auto& camera = problem.cameraParameters[observation.cameraIndex];
            const auto& point = problem.tiePoints[observation.pointIndex];
            if (!problem.cameraModels.empty())
            {
                ProjectionState state;
                ProjectionEvaluation evaluation;
                const CameraModel& model = problem.cameraModels[observation.cameraIndex];
                if (!prepareProjectionState(model, observation.linePixels, &state) ||
                    !evaluateProjection(model, state, observation, camera, point, &evaluation))
                {
                    return false;
                }
                const double inverse_sigma = 1.0 / options.imageSigmaPixels;
                for (int row = 0; row < 2; ++row)
                {
                    residual[row] = evaluation.residualPixels[row] * inverse_sigma;
                    for (int column = 0; column < 6; ++column)
                    {
                        cameraJacobian[row * 6 + column] = evaluation.cameraJacobian[row * 6 + column] * inverse_sigma;
                    }
                    for (int column = 0; column < 3; ++column)
                    {
                        pointJacobian[row * 3 + column] = evaluation.pointJacobian[row * 3 + column] * inverse_sigma;
                    }
                }
                return true;
            }
            if (!evaluateImageObservation(problem, observation, camera, point, options.imageSigmaPixels, residual))
            {
                return false;
            }
            for (int block = 0; block < 2; ++block)
            {
                const int blockSize = block == 0 ? 6 : 3;
                double* jacobian = block == 0 ? cameraJacobian : pointJacobian;
                for (int column = 0; column < blockSize; ++column)
                {
                    const double step = block == 1 ? options.finiteDifferencePointStepMeters
                                                   : (column < 3 ? options.finiteDifferencePositionStepMeters
                                                                 : options.finiteDifferenceAngleStepRadians);
                    auto plusCamera = camera;
                    auto minusCamera = camera;
                    auto plusPoint = point;
                    auto minusPoint = point;
                    if (block == 0)
                    {
                        plusCamera[column] += step;
                        minusCamera[column] -= step;
                    }
                    else
                    {
                        plusPoint[column] += step;
                        minusPoint[column] -= step;
                    }
                    double plusResidual[2]{};
                    double minusResidual[2]{};
                    if (!evaluateImageObservation(
                            problem, observation, plusCamera, plusPoint, options.imageSigmaPixels, plusResidual) ||
                        !evaluateImageObservation(
                            problem, observation, minusCamera, minusPoint, options.imageSigmaPixels, minusResidual))
                    {
                        return false;
                    }
                    for (int row = 0; row < 2; ++row)
                    {
                        jacobian[row * blockSize + column] = (plusResidual[row] - minusResidual[row]) / (2.0 * step);
                    }
                }
            }
            return true;
        }

        bool linearizeLaserRange(const std::array<double, 6>& camera,
                                 const Vector3& point,
                                 const LaserObservation& observation,
                                 double* residual,
                                 double* cameraJacobian,
                                 double* pointJacobian)
        {
            const Vector3 delta{{point[0] - observation.nominalSensorCenterMeters[0] - camera[0],
                                 point[1] - observation.nominalSensorCenterMeters[1] - camera[1],
                                 point[2] - observation.nominalSensorCenterMeters[2] - camera[2]}};
            const double range = std::hypot(std::hypot(delta[0], delta[1]), delta[2]);
            if (!(range > 1.0e-12) || !std::isfinite(range))
            {
                return false;
            }
            const double inverseSigma = 1.0 / observation.sigmaMeters;
            *residual = (range - observation.observedRangeMeters) * inverseSigma;
            std::fill_n(cameraJacobian, 6, 0.0);
            for (int axis = 0; axis < 3; ++axis)
            {
                pointJacobian[axis] = delta[axis] * inverseSigma / range;
                cameraJacobian[axis] = -pointJacobian[axis];
            }
            return std::isfinite(*residual);
        }

        void addPointPrior(const LaserPoint& laserPoint,
                           int laserBlock,
                           plamatrix::internal::BlockNormalEquations<double>* equations)
        {
            double residual[3]{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    residual[row] += laserPoint.sqrtInformation[row * 3 + column] *
                                     (laserPoint.refinedMeters[column] - laserPoint.initialMeters[column]);
                }
            }
            equations->addEliminatedResidualBlock(laserBlock, laserPoint.sqrtInformation.data(), residual, 3);
        }

        double evaluateImageRange(const Problem& problem, const Options& options, std::size_t begin, std::size_t end)
        {
            const double imageDelta = options.imageHuberDeltaPixels / options.imageSigmaPixels;
            double cost = 0.0;
            for (std::size_t index = begin; index < end; ++index)
            {
                const auto& observation = problem.imageObservations[index];
                double residual[2]{};
                if (!evaluateImageObservation(problem,
                                              observation,
                                              problem.cameraParameters[observation.cameraIndex],
                                              problem.tiePoints[observation.pointIndex],
                                              options.imageSigmaPixels,
                                              residual))
                {
                    return std::numeric_limits<double>::infinity();
                }
                cost += plamatrix::internal::evaluateHuberLoss(residual[0] * residual[0] + residual[1] * residual[1],
                                                               imageDelta)
                            .cost;
            }
            return cost;
        }

        void assembleImageRange(const Problem& problem,
                                const Options& options,
                                std::size_t begin,
                                std::size_t end,
                                plamatrix::internal::BlockNormalEquations<double>* equations)
        {
            const double imageDelta = options.imageHuberDeltaPixels / options.imageSigmaPixels;
            for (std::size_t index = begin; index < end; ++index)
            {
                const auto& observation = problem.imageObservations[index];
                double residual[2]{};
                double cameraJacobian[12]{};
                double pointJacobian[6]{};
                if (!linearizeImageObservation(problem, observation, options, residual, cameraJacobian, pointJacobian))
                {
                    throw std::runtime_error("line-scan image projection failed during linearization");
                }
                const auto robust = plamatrix::internal::evaluateHuberLoss(
                    residual[0] * residual[0] + residual[1] * residual[1], imageDelta);
                equations->addResidualBlock(observation.cameraIndex,
                                            observation.pointIndex,
                                            cameraJacobian,
                                            pointJacobian,
                                            residual,
                                            2,
                                            robust.weight);
            }
        }

    } // namespace

    std::vector<int> makeLaserBlocks(const Problem& problem)
    {
        std::vector<int> blocks(problem.laserPoints.size(), -1);
        int nextBlock = static_cast<int>(problem.tiePoints.size());
        for (std::size_t index = 0; index < problem.laserPoints.size(); ++index)
        {
            if (problem.laserPoints[index].mode == LaserPointMode::Constrained)
            {
                blocks[index] = nextBlock++;
            }
        }
        return blocks;
    }

    double evaluateObjective(const Problem& problem, const Options& options)
    {
        const int requestedThreads = options.threadCount > 0 ? options.threadCount : maxThreads();
        const int threadCount =
            std::min<int>(std::max(1, requestedThreads), static_cast<int>(problem.imageObservations.size()));
        double cost = 0.0;
        if (threadCount <= 1 || problem.imageObservations.size() < 128)
        {
            cost = evaluateImageRange(problem, options, 0, problem.imageObservations.size());
        }
        else
        {
            std::vector<double> partialCosts(static_cast<std::size_t>(threadCount), 0.0);
#ifdef PLABUNDLE_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadCount) schedule(static, 1)
#endif
            for (int thread = 0; thread < threadCount; ++thread)
            {
                const std::size_t begin = problem.imageObservations.size() * static_cast<std::size_t>(thread) /
                                          static_cast<std::size_t>(threadCount);
                const std::size_t end = problem.imageObservations.size() * static_cast<std::size_t>(thread + 1) /
                                        static_cast<std::size_t>(threadCount);
                partialCosts[static_cast<std::size_t>(thread)] = evaluateImageRange(problem, options, begin, end);
            }
            cost = std::accumulate(partialCosts.begin(), partialCosts.end(), 0.0);
        }
        const double angleSigma = options.cameraAngleSigmaDegrees * std::acos(-1.0) / 180.0;
        for (const auto& camera : problem.cameraParameters)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                cost += 0.5 * std::pow(camera[axis] / options.cameraPositionSigmaMeters, 2.0);
                cost += 0.5 * std::pow(camera[axis + 3] / angleSigma, 2.0);
            }
        }
        if (!options.enableLaserRangeConstraints)
        {
            return cost;
        }
        for (const auto& observation : problem.laserObservations)
        {
            const auto& laserPoint = problem.laserPoints[observation.laserPointIndex];
            double residual = 0.0;
            double cameraJacobian[6]{};
            double pointJacobian[3]{};
            if (!linearizeLaserRange(problem.cameraParameters[observation.cameraIndex],
                                     laserPoint.refinedMeters,
                                     observation,
                                     &residual,
                                     cameraJacobian,
                                     pointJacobian))
            {
                return std::numeric_limits<double>::infinity();
            }
            cost += options.laserRangeWeight *
                    plamatrix::internal::evaluateHuberLoss(residual * residual, options.laserRangeHuberDeltaSigma).cost;
            if (laserPoint.mode == LaserPointMode::Constrained)
            {
                double prior[3]{};
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        prior[row] += laserPoint.sqrtInformation[row * 3 + column] *
                                      (laserPoint.refinedMeters[column] - laserPoint.initialMeters[column]);
                    }
                    cost += 0.5 * prior[row] * prior[row];
                }
            }
        }
        return cost;
    }

    plamatrix::internal::BlockNormalEquations<double>
    buildEquations(const Problem& problem, const Options& options, const std::vector<int>& laserBlocks)
    {
        const int constrainedLaserCount = static_cast<int>(
            std::count_if(laserBlocks.begin(), laserBlocks.end(), [](int block) { return block >= 0; }));
        plamatrix::internal::BlockNormalEquations<double> equations(
            problem.cameraParameters.size(), problem.tiePoints.size() + constrainedLaserCount, 6, 3);
        const int requestedThreads = options.threadCount > 0 ? options.threadCount : maxThreads();
        const int threadCount =
            std::min<int>(std::max(1, requestedThreads), static_cast<int>(problem.imageObservations.size()));
        if (threadCount <= 1 || problem.imageObservations.size() < 128)
        {
            assembleImageRange(problem, options, 0, problem.imageObservations.size(), &equations);
        }
        else
        {
            std::vector<std::unique_ptr<plamatrix::internal::BlockNormalEquations<double>>> partials;
            std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threadCount));
            partials.reserve(static_cast<std::size_t>(threadCount));
            for (int thread = 0; thread < threadCount; ++thread)
            {
                partials.push_back(std::make_unique<plamatrix::internal::BlockNormalEquations<double>>(
                    problem.cameraParameters.size(), problem.tiePoints.size() + constrainedLaserCount, 6, 3));
            }
#ifdef PLABUNDLE_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadCount) schedule(static, 1)
#endif
            for (int thread = 0; thread < threadCount; ++thread)
            {
                const std::size_t begin = problem.imageObservations.size() * static_cast<std::size_t>(thread) /
                                          static_cast<std::size_t>(threadCount);
                const std::size_t end = problem.imageObservations.size() * static_cast<std::size_t>(thread + 1) /
                                        static_cast<std::size_t>(threadCount);
                try
                {
                    assembleImageRange(problem, options, begin, end, partials[static_cast<std::size_t>(thread)].get());
                }
                catch (...)
                {
                    errors[static_cast<std::size_t>(thread)] = std::current_exception();
                }
            }
            for (int thread = 0; thread < threadCount; ++thread)
            {
                const auto index = static_cast<std::size_t>(thread);
                if (errors[index])
                {
                    std::rethrow_exception(errors[index]);
                }
                equations.mergeFrom(*partials[index]);
            }
        }
        const double angleSigma = options.cameraAngleSigmaDegrees * std::acos(-1.0) / 180.0;
        for (std::size_t index = 0; index < problem.cameraParameters.size(); ++index)
        {
            double residual[6]{};
            double jacobian[36]{};
            for (int axis = 0; axis < 6; ++axis)
            {
                const double inverseSigma = 1.0 / (axis < 3 ? options.cameraPositionSigmaMeters : angleSigma);
                residual[axis] = problem.cameraParameters[index][axis] * inverseSigma;
                jacobian[axis * 6 + axis] = inverseSigma;
            }
            equations.addPrimaryResidualBlock(index, jacobian, residual, 6);
        }
        if (!options.enableLaserRangeConstraints)
        {
            return equations;
        }
        for (const auto& observation : problem.laserObservations)
        {
            const auto& laserPoint = problem.laserPoints[observation.laserPointIndex];
            double residual = 0.0;
            double cameraJacobian[6]{};
            double pointJacobian[3]{};
            if (!linearizeLaserRange(problem.cameraParameters[observation.cameraIndex],
                                     laserPoint.refinedMeters,
                                     observation,
                                     &residual,
                                     cameraJacobian,
                                     pointJacobian))
            {
                throw std::runtime_error("line-scan laser range is numerically invalid");
            }
            const auto robust =
                plamatrix::internal::evaluateHuberLoss(residual * residual, options.laserRangeHuberDeltaSigma);
            const double weight = options.laserRangeWeight * robust.weight;
            const int laserBlock = laserBlocks[observation.laserPointIndex];
            if (laserBlock >= 0)
            {
                equations.addResidualBlock(
                    observation.cameraIndex, laserBlock, cameraJacobian, pointJacobian, &residual, 1, weight);
                addPointPrior(laserPoint, laserBlock, &equations);
            }
            else
            {
                equations.addPrimaryResidualBlock(observation.cameraIndex, cameraJacobian, &residual, 1, weight);
            }
        }
        return equations;
    }

    double maximumStepNorm(const std::vector<double>& primaryStep, const std::vector<double>& eliminatedStep)
    {
        double maximum = 0.0;
        for (double value : primaryStep)
        {
            maximum = std::max(maximum, std::abs(value));
        }
        for (double value : eliminatedStep)
        {
            maximum = std::max(maximum, std::abs(value));
        }
        return maximum;
    }

    void applyStep(const std::vector<int>& laserBlocks,
                   const std::vector<double>& primaryStep,
                   const std::vector<double>& eliminatedStep,
                   Problem* problem)
    {
        for (std::size_t block = 0; block < problem->cameraParameters.size(); ++block)
        {
            for (int axis = 0; axis < 6; ++axis)
            {
                problem->cameraParameters[block][axis] += primaryStep[block * 6 + axis];
            }
        }
        for (std::size_t block = 0; block < problem->tiePoints.size(); ++block)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                problem->tiePoints[block][axis] += eliminatedStep[block * 3 + axis];
            }
        }
        for (std::size_t index = 0; index < laserBlocks.size(); ++index)
        {
            if (laserBlocks[index] < 0)
            {
                continue;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                problem->laserPoints[index].refinedMeters[axis] += eliminatedStep[laserBlocks[index] * 3 + axis];
            }
        }
    }

} // namespace plabundle::linescan::detail::assembly
