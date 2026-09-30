#pragma once

#include <plabundle/backend.h>
#include <plabundle/options.h>
#include <plabundle/problem.h>
#include <plabundle/result.h>

#include <memory>

namespace plabundle
{

    /// Reusable numerical workspace for sequential bundle solves. The caller owns its lifetime.
    /// A workspace must not be used by concurrent solve calls.
    class SolverWorkspace final
    {
    public:
        SolverWorkspace();
        ~SolverWorkspace();
        SolverWorkspace(SolverWorkspace&&) noexcept;
        SolverWorkspace& operator=(SolverWorkspace&&) noexcept;
        SolverWorkspace(const SolverWorkspace&) = delete;
        SolverWorkspace& operator=(const SolverWorkspace&) = delete;

        /// Release cached numerical and symbolic data.
        void clear() noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
        friend class Solver;
    };

    class Solver final
    {
    public:
        Result solve(const Problem& problem) const;
        Result solve(const Problem& problem, const SolveOptions& options) const;
        Result solve(const Problem& problem, const Options& options) const;
        Result solve(const Problem& problem, const SolveOptions& options, SolverWorkspace& workspace) const;
        Result solve(const Problem& problem, const Options& options, SolverWorkspace& workspace) const;

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
