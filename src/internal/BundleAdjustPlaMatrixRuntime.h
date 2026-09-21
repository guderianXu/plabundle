#pragma once

#include "BundleAdjustTypes.h"

#include <plamatrix/optimization/block_schur.h>

namespace plabundle::internal
{

    plamatrix::SchurComplementLinearBackend plaMatrixLinearBackend(BABackend backend);

    const char* plaMatrixLinearBackendName(plamatrix::SchurComplementLinearBackend backend);

} // namespace plabundle::internal
