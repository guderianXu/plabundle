#pragma once

#include "BundleAdjustTypes.h"

#include <plamatrix/internal/optimization/block_schur.h>

namespace plabundle::internal
{

    plamatrix::internal::SchurComplementLinearBackend plaMatrixLinearBackend(BABackend backend);

    const char* plaMatrixLinearBackendName(plamatrix::internal::SchurComplementLinearBackend backend);

} // namespace plabundle::internal
