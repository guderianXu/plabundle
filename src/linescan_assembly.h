#pragma once

#include "linescan_internal.h"

#include <plamatrix/internal/optimization/block_schur.h>

#include <vector>

namespace plabundle::linescan::detail::assembly
{

    std::vector<int> makeLaserBlocks(const Problem& problem);
    double evaluateObjective(const Problem& problem, const Options& options);
    plamatrix::internal::BlockNormalEquations<double>
    buildEquations(const Problem& problem, const Options& options, const std::vector<int>& laserBlocks);
    double maximumStepNorm(const std::vector<double>& primaryStep, const std::vector<double>& eliminatedStep);
    void applyStep(const std::vector<int>& laserBlocks,
                   const std::vector<double>& primaryStep,
                   const std::vector<double>& eliminatedStep,
                   Problem* problem);

} // namespace plabundle::linescan::detail::assembly
