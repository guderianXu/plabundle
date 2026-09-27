#pragma once

#include <plabundle/constraints.h>

#include <array>
#include <string>

namespace plabundle::internal
{

    struct PosePriorWhitening
    {
        int residualSize = 0;
        std::array<int, 6> componentIndices{};
        // Compact residual rows, canonical [rotation, position] columns.
        std::array<double, 36> matrix{};
    };

    bool
    makePosePriorWhitening(const CameraPosePrior& prior, PosePriorWhitening* whitening, std::string* error = nullptr);

} // namespace plabundle::internal
