#pragma once

#include "BundleAdjustOptions.h"
#include "BundleAdjustResult.h"

#include <plabundle/problem.h>
#include <plabundle/result.h>

#include <string>

namespace plabundle::internal
{

    BAOptions makeSolverOptions(const Problem& problem, const Options& options, Backend usedBackend);
    Result makePublicResult(const BAResult& source,
                            Backend requestedBackend,
                            Backend usedBackend,
                            const std::string& selectionReason = {});

} // namespace plabundle::internal
