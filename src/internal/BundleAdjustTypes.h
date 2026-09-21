#pragma once

#include <plabundle/backend.h>
#include <plabundle/options.h>
#include <plabundle/problem.h>

#include <cstddef>
#include <string>

namespace plabundle::internal
{

    using BABackend = Backend;
    using BASolveStatus = SolveStatus;
    using BABackendCapabilities = BackendCapabilities;
    using BAGaugePolicy = GaugePolicy;
    using BAIntrinsicParameter = IntrinsicParameter;
    using BAIntrinsicParameterMask = IntrinsicParameterMask;
    inline constexpr std::size_t kBAIntrinsicParameterCount = kIntrinsicParameterCount;

    using BAProblemStats = ProblemStats;

    struct BABackendDecision
    {
        BABackend backend = BABackend::PlaMatrixCpu;
        std::string reason;
    };

} // namespace plabundle::internal
