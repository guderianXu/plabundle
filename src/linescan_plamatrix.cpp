#include "linescan_internal.h"
#include "linescan_assembly.h"

#include <plamatrix/internal/optimization/levenberg_marquardt.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace plabundle::linescan::detail
{
    namespace
    {

        void setError(std::string* errorMessage, const std::string& message)
        {
            if (errorMessage)
            {
                *errorMessage = message;
            }
        }

        plamatrix::internal::SchurComplementLinearBackend linearBackend(plabundle::Backend backend)
        {
            if (backend == plabundle::Backend::PlaMatrixCuda)
            {
                return plamatrix::internal::SchurComplementLinearBackend::Cuda;
            }
            if (backend == plabundle::Backend::PlaMatrixOpenCl)
            {
                return plamatrix::internal::SchurComplementLinearBackend::OpenCl;
            }
            if (backend == plabundle::Backend::PlaMatrixVulkan)
            {
                return plamatrix::internal::SchurComplementLinearBackend::Vulkan;
            }
            return plamatrix::internal::hasSparseDirectSchurSolver()
                       ? plamatrix::internal::SchurComplementLinearBackend::SparseCpu
                       : plamatrix::internal::SchurComplementLinearBackend::Cpu;
        }

        const char* linearBackendName(plamatrix::internal::SchurComplementLinearBackend backend)
        {
            switch (backend)
            {
            case plamatrix::internal::SchurComplementLinearBackend::Cpu:
                return "block_jacobi_pcg_cpu";
            case plamatrix::internal::SchurComplementLinearBackend::DenseCpu:
                return "dense_cholesky_cpu";
            case plamatrix::internal::SchurComplementLinearBackend::SparseCpu:
                return "sparse_cholesky_native_cpu";
            case plamatrix::internal::SchurComplementLinearBackend::Cuda:
                return "block_jacobi_pcg_cuda";
            case plamatrix::internal::SchurComplementLinearBackend::OpenCl:
                return "block_jacobi_pcg_opencl";
            case plamatrix::internal::SchurComplementLinearBackend::Vulkan:
                return "block_jacobi_pcg_vulkan";
            }
            return "unknown";
        }

    } // namespace

    bool solvePlaMatrix(Problem* problem, const Options& options, Result* result, std::string* errorMessage)
    {
        if (!problem || !result)
        {
            setError(errorMessage, "invalid planetary line-scan PlaMatrix working set");
            return false;
        }
        auto backend = linearBackend(options.backend);
        if (backend == plamatrix::internal::SchurComplementLinearBackend::Cpu &&
            problem->cameraParameters.size() <= static_cast<std::size_t>(std::max(1, options.maxDenseSchurCameras)))
        {
            backend = plamatrix::internal::SchurComplementLinearBackend::DenseCpu;
        }
        result->linearSolverName = linearBackendName(backend);
        const Problem initialWorkingSet = *problem;
        const auto isCancelled = [&options]()
        { return options.cancelFlag && options.cancelFlag->load(std::memory_order_relaxed); };
        if (isCancelled())
        {
            result->terminationType = "CANCELLED";
            result->backendMessage = "PlaMatrix line-scan BA cancelled before solving";
            setError(errorMessage, result->backendMessage);
            return false;
        }
        const std::vector<int> laserBlocks = options.enableLaserRangeConstraints
                                                 ? assembly::makeLaserBlocks(*problem)
                                                 : std::vector<int>(problem->laserPoints.size(), -1);
        double currentCost = assembly::evaluateObjective(*problem, options);
        plamatrix::internal::LevenbergMarquardtStrategy<double> lm;
        plamatrix::internal::SchurComplementSolverWorkspace<double> workspace;
        bool converged = false;
        try
        {
            int iteration = 0;
            while (iteration < options.maximumIterations && !converged)
            {
                if (isCancelled())
                {
                    *problem = initialWorkingSet;
                    result->terminationType = "CANCELLED";
                    result->solutionUsable = false;
                    result->backendMessage = "PlaMatrix line-scan BA cancelled without publishing a partial solution";
                    setError(errorMessage, result->backendMessage);
                    return false;
                }
                const auto equations = assembly::buildEquations(*problem, options, laserBlocks);
                bool relinearize = false;
                std::vector<double> retryPrimaryStep;
                while (iteration < options.maximumIterations && !converged && !relinearize)
                {
                    if (isCancelled())
                    {
                        *problem = initialWorkingSet;
                        result->terminationType = "CANCELLED";
                        result->solutionUsable = false;
                        result->backendMessage =
                            "PlaMatrix line-scan BA cancelled without publishing a partial solution";
                        setError(errorMessage, result->backendMessage);
                        return false;
                    }
                    plamatrix::internal::SchurComplementSolverOptions<double> solverOptions;
                    solverOptions.linearBackend = backend;
                    solverOptions.deviceIndex = options.plaMatrixDevice;
                    solverOptions.useMixedPrecision =
                        backend == plamatrix::internal::SchurComplementLinearBackend::Vulkan;
                    solverOptions.maxIterations =
                        (backend == plamatrix::internal::SchurComplementLinearBackend::Cpu ||
                         backend == plamatrix::internal::SchurComplementLinearBackend::DenseCpu ||
                         backend == plamatrix::internal::SchurComplementLinearBackend::SparseCpu)
                            ? std::max(100, static_cast<int>(problem->cameraParameters.size()) * 24)
                            : std::max(200, static_cast<int>(problem->cameraParameters.size()) * 30);
                    solverOptions.relativeTolerance = iteration < 2 ? 1.0e-3 : (iteration < 4 ? 1.0e-5 : 1.0e-8);
                    if (iteration + 1 >= options.maximumIterations)
                    {
                        solverOptions.relativeTolerance = 1.0e-10;
                    }
                    solverOptions.absoluteTolerance = 1.0e-12;
                    solverOptions.useInitialGuess = !retryPrimaryStep.empty();
                    std::vector<double> primaryStep = retryPrimaryStep;
                    std::vector<double> eliminatedStep;
                    auto report = plamatrix::internal::solveDampedSchurComplement(
                        equations, lm.damping(), solverOptions, workspace, &primaryStep, &eliminatedStep);
                    if (!report.converged && (backend == plamatrix::internal::SchurComplementLinearBackend::DenseCpu ||
                                              backend == plamatrix::internal::SchurComplementLinearBackend::SparseCpu))
                    {
                        backend = plamatrix::internal::SchurComplementLinearBackend::Cpu;
                        result->linearSolverName = linearBackendName(backend);
                        solverOptions.linearBackend = backend;
                        solverOptions.useInitialGuess = false;
                        primaryStep.clear();
                        eliminatedStep.clear();
                        report = plamatrix::internal::solveDampedSchurComplement(
                            equations, lm.damping(), solverOptions, workspace, &primaryStep, &eliminatedStep);
                    }
                    if (isCancelled())
                    {
                        *problem = initialWorkingSet;
                        result->terminationType = "CANCELLED";
                        result->solutionUsable = false;
                        result->backendMessage =
                            "PlaMatrix line-scan BA cancelled without publishing a partial solution";
                        setError(errorMessage, result->backendMessage);
                        return false;
                    }
                    ++iteration;
                    ++result->iterations;
                    if (!report.deviceName.empty())
                    {
                        result->deviceName = report.deviceName;
                        result->usedGpu = true;
                    }
                    if (!report.converged)
                    {
                        retryPrimaryStep = primaryStep;
                        lm.rejectStep();
                        continue;
                    }
                    if (assembly::maximumStepNorm(primaryStep, eliminatedStep) <= 1.0e-10)
                    {
                        converged = true;
                        break;
                    }
                    auto candidate = *problem;
                    assembly::applyStep(laserBlocks, primaryStep, eliminatedStep, &candidate);
                    const double candidateCost = assembly::evaluateObjective(candidate, options);
                    if (std::isfinite(candidateCost) && candidateCost < currentCost)
                    {
                        const double relativeChange =
                            (currentCost - candidateCost) / std::max(1.0, std::abs(currentCost));
                        *problem = std::move(candidate);
                        currentCost = candidateCost;
                        lm.acceptStep();
                        converged = relativeChange <= 1.0e-12;
                        relinearize = !converged;
                    }
                    else
                    {
                        retryPrimaryStep = primaryStep;
                        lm.rejectStep();
                    }
                }
            }
        }
        catch (const std::exception& error)
        {
            setError(errorMessage, std::string("PlaMatrix line-scan BA failed: ") + error.what());
            result->terminationType = "NUMERICAL_FAILURE";
            result->backendMessage = errorMessage ? *errorMessage : error.what();
            return false;
        }
        result->converged = converged;
        result->solutionUsable =
            std::isfinite(currentCost) && (converged || lm.acceptedSteps() > 0 || currentCost <= 1.0e-20);
        result->terminationType = converged ? "CONVERGENCE" : "NO_CONVERGENCE";
        result->solverBriefReport = "PlaMatrix " + result->linearSolverName +
                                    ": accepted=" + std::to_string(lm.acceptedSteps()) +
                                    ", rejected=" + std::to_string(lm.rejectedSteps());
        result->message = converged ? "planetary line-scan PlaMatrix bundle adjustment converged"
                                    : "planetary line-scan PlaMatrix bundle adjustment reached the iteration limit";
        result->backendMessage = result->message;
        if (!result->solutionUsable)
        {
            setError(errorMessage, "PlaMatrix line-scan BA did not produce a usable solution");
            return false;
        }
        return true;
    }

} // namespace plabundle::linescan::detail
