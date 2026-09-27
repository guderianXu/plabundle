#pragma once

#include <plabundle/backend.h>
#include <plabundle/options.h>
#include <plabundle/problem.h>
#include <plabundle/result.h>

namespace plabundle
{

    class Solver final
    {
    public:
        Result solve(const Problem& problem) const;
        Result solve(const Problem& problem, const SolveOptions& options) const;
        Result solve(const Problem& problem, const Options& options) const;

        static bool isBackendAvailable(Backend backend, int deviceIndex = 0) noexcept;
        static BackendCapabilities backendCapabilities(Backend backend) noexcept;
        static bool
        autoBackendMeetsScaleThreshold(Backend backend, const ProblemStats& stats, const Options& options) noexcept;
        static bool autoBackendMeetsScaleThreshold(Backend backend,
                                                   const ProblemStats& stats,
                                                   const SolveOptions& options) noexcept;
        static BackendDecision decideBackendForProblem(const Problem& problem, const Options& options);
        static BackendDecision decideBackendForProblem(const Problem& problem, const SolveOptions& options);
        static Backend selectBackendForProblem(const Problem& problem, const Options& options);
        static Backend selectBackendForProblem(const Problem& problem, const SolveOptions& options);
    };

} // namespace plabundle
