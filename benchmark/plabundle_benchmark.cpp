#include <plabundle/solver.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct BenchmarkSettings
    {
        int cameraCount = 80;
        int trackCount = 3000;
        int viewsPerTrack = 8;
        int iterations = 8;
        int threadCount = 0;
        int repetitions = 3;
        int deviceIndex = 0;
        bool mixedPrecision = false;
        std::string backends = "plamatrix_cpu,plamatrix_cuda,plamatrix_vulkan,plamatrix_opencl,auto";
    };

    plabundle::FrameCamera makeCamera(double centerX, double centerY, double centerZ)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter = {centerX, centerY, centerZ};
        camera.focalXPixels = 1000.0;
        camera.focalYPixels = 1000.0;
        camera.principalXPixel = 512.0;
        camera.principalYPixel = 384.0;
        return camera;
    }

    std::vector<plabundle::FrameCamera> makeCameras(int count)
    {
        std::vector<plabundle::FrameCamera> cameras;
        cameras.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            const double position = (static_cast<double>(index) / std::max(1, count - 1) - 0.5) * 18.0;
            cameras.push_back(makeCamera(position, std::sin(index * 0.45) * 3.0, 0.0));
        }
        return cameras;
    }

    std::vector<plabundle::Track>
    makeTracks(const std::vector<plabundle::FrameCamera>& cameras, int trackCount, int viewsPerTrack)
    {
        std::mt19937 generator(7U);
        std::uniform_real_distribution<double> xy_distribution(-4.0, 4.0);
        std::uniform_real_distribution<double> z_distribution(32.0, 52.0);
        std::normal_distribution<double> initial_noise(0.0, 0.30);
        std::normal_distribution<double> image_noise(0.0, 0.05);

        std::vector<plabundle::Track> tracks;
        tracks.reserve(static_cast<std::size_t>(trackCount));
        for (int track_index = 0; track_index < trackCount; ++track_index)
        {
            const std::array<double, 3> truth{
                {xy_distribution(generator), xy_distribution(generator), z_distribution(generator)}};
            plabundle::Track track;
            track.initialPoint = {truth[0] + initial_noise(generator),
                                  truth[1] + initial_noise(generator),
                                  truth[2] + initial_noise(generator)};
            const int first_camera = track_index % static_cast<int>(cameras.size());
            for (int view_index = 0; view_index < viewsPerTrack; ++view_index)
            {
                const int camera_index = (first_camera + view_index) % static_cast<int>(cameras.size());
                plabundle::Projection projection;
                if (plabundle::projectWorldPoint(cameras[static_cast<std::size_t>(camera_index)], truth, &projection))
                {
                    track.observations.push_back({camera_index,
                                                  projection.pixel[0] + image_noise(generator),
                                                  projection.pixel[1] + image_noise(generator),
                                                  1.0,
                                                  1.0});
                }
            }
            if (track.observations.size() >= 2)
            {
                tracks.push_back(std::move(track));
            }
        }
        return tracks;
    }

    plabundle::Problem makeProblem(const BenchmarkSettings& settings)
    {
        const std::vector<plabundle::FrameCamera> truth_cameras = makeCameras(settings.cameraCount);
        plabundle::Problem problem;
        problem.cameras = truth_cameras;
        problem.tracks = makeTracks(truth_cameras, settings.trackCount, settings.viewsPerTrack);
        for (std::size_t index = 2; index < problem.cameras.size(); ++index)
        {
            const double sign = index % 2 == 0 ? 1.0 : -1.0;
            const double scale = 1.0 + static_cast<double>(index % 5) * 0.1;
            if (!plabundle::applyPoseDelta(&problem.cameras[index],
                                           {0.0007 * sign * scale,
                                            -0.0005 * sign,
                                            0.0004 * scale,
                                            0.025 * sign * scale,
                                            -0.018 * sign,
                                            0.012 * scale}))
            {
                throw std::runtime_error("failed to perturb a benchmark camera pose");
            }
        }
        problem.fixedCameraIndices = {0, 1};
        problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;
        return problem;
    }

    int parseInteger(const char* text, const char* name, int minimum)
    {
        std::size_t parsed = 0;
        const long long value = std::stoll(text, &parsed);
        if (parsed != std::string(text).size() || value < minimum || value > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument(std::string(name) + " must be an integer >= " + std::to_string(minimum));
        }
        return static_cast<int>(value);
    }

    bool parseBoolean(const char* text, const char* name)
    {
        const std::string value = text;
        if (value == "0")
        {
            return false;
        }
        if (value == "1")
        {
            return true;
        }
        throw std::invalid_argument(std::string(name) + " must be 0 or 1");
    }

    std::vector<std::string> splitBackends(const std::string& raw)
    {
        std::vector<std::string> names;
        std::stringstream stream(raw);
        std::string name;
        while (std::getline(stream, name, ','))
        {
            name.erase(std::remove_if(name.begin(),
                                      name.end(),
                                      [](unsigned char character) { return std::isspace(character) != 0; }),
                       name.end());
            if (!name.empty())
            {
                names.push_back(name);
            }
        }
        if (names.empty())
        {
            throw std::invalid_argument("at least one backend is required");
        }
        return names;
    }

    plabundle::Backend parseBackend(const std::string& name)
    {
        if (name == "auto")
        {
            return plabundle::Backend::Auto;
        }
        if (name == "plamatrix_cpu")
        {
            return plabundle::Backend::PlaMatrixCpu;
        }
        if (name == "plamatrix_cuda")
        {
            return plabundle::Backend::PlaMatrixCuda;
        }
        if (name == "plamatrix_opencl")
        {
            return plabundle::Backend::PlaMatrixOpenCl;
        }
        if (name == "plamatrix_vulkan")
        {
            return plabundle::Backend::PlaMatrixVulkan;
        }
        throw std::invalid_argument("unknown backend: " + name);
    }

    std::string sanitizeField(std::string value)
    {
        std::replace(value.begin(), value.end(), ',', ';');
        std::replace(value.begin(), value.end(), '\n', ' ');
        std::replace(value.begin(), value.end(), '\r', ' ');
        return value;
    }

    void printUsage(const char* program)
    {
        std::cout << "usage: " << program
                  << " [cameras tracks views iterations threads repetitions backends device mixed_precision]\n"
                  << "  threads: 0 selects the runtime default\n"
                  << "  backends: comma-separated auto/plamatrix_cpu/plamatrix_cuda/plamatrix_vulkan/"
                     "plamatrix_opencl\n"
                  << "  mixed_precision: 0 or 1; applies to accelerated backends\n";
    }

    int runBackend(const plabundle::Problem& problem, const BenchmarkSettings& settings, plabundle::Backend backend)
    {
        plabundle::SolveOptions options;
        options.backend.requested = backend;
        options.solver.maxIterations = settings.iterations;
        options.solver.numThreads = settings.threadCount;
        options.calibration.refineCameraPose = true;
        options.solver.enablePointFilter = false;
        options.solver.logIterationProgress = false;
        options.backend.plaMatrixDevice = settings.deviceIndex;
        options.backend.enablePlaMatrixMixedPrecision = settings.mixedPrecision;
        options.backend.allowFallback = false;

        const bool runtime_available = plabundle::Solver::isBackendAvailable(backend, settings.deviceIndex);
        int failures = 0;
        for (int repetition = 1; repetition <= settings.repetitions; ++repetition)
        {
            const auto start = std::chrono::steady_clock::now();
            const plabundle::Result result = plabundle::Solver().solve(problem, options);
            const double wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            std::cout << "run,backend=" << plabundle::backendName(backend) << ",repetition=" << repetition
                      << ",phase=" << (repetition == 1 ? "cold" : "warm")
                      << ",available=" << (runtime_available ? "true" : "false")
                      << ",requested=" << plabundle::backendName(result.requestedBackend)
                      << ",used=" << plabundle::backendName(result.usedBackend)
                      << ",status=" << plabundle::solveStatusName(result.status)
                      << ",usable=" << (result.usable() ? "true" : "false")
                      << ",gpu=" << (result.usedGpu ? "true" : "false")
                      << ",fallback=" << (result.backendFallback ? "true" : "false")
                      << ",selection_reason=" << sanitizeField(result.backendSelectionReason)
                      << ",backend_message=" << sanitizeField(result.backendMessage)
                      << ",quality_rejected=" << (result.qualityGateRejected ? "true" : "false")
                      << ",valid_ratio=" << result.quality.validTrackRatio
                      << ",rms_before=" << result.quality.meanRmsBefore << ",rms_after=" << result.quality.meanRmsAfter
                      << ",initial_cost=" << result.plaMatrix.initialCost
                      << ",final_cost=" << result.plaMatrix.finalCost
                      << ",linear_solver=" << result.plaMatrix.linearSolverName
                      << ",device=" << sanitizeField(result.plaMatrix.deviceName)
                      << ",linear_iterations=" << result.plaMatrix.linearIterations
                      << ",schur_pattern_builds=" << result.plaMatrix.schurPatternBuilds
                      << ",schur_pattern_reuses=" << result.plaMatrix.schurPatternReuses
                      << ",schur_on_device=" << (result.plaMatrix.schurAssemblyOnDevice ? "true" : "false")
                      << ",mixed_precision_used=" << (result.plaMatrix.mixedPrecisionUsed ? "true" : "false")
                      << ",assembly_seconds=" << result.plaMatrix.assemblySeconds
                      << ",linear_solve_seconds=" << result.plaMatrix.linearSolveSeconds
                      << ",back_substitution_seconds=" << result.plaMatrix.backSubstitutionSeconds
                      << ",solve_seconds=" << result.timing.solveSeconds
                      << ",total_seconds=" << result.timing.totalSeconds << ",wall_seconds=" << wall_seconds << '\n';

            if (runtime_available && !result.usable())
            {
                ++failures;
            }
        }
        return failures;
    }

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
        {
            printUsage(argv[0]);
            return 0;
        }
        if (argc > 10)
        {
            throw std::invalid_argument("too many benchmark arguments; use --help for the supported syntax");
        }

        BenchmarkSettings settings;
        settings.cameraCount = argc > 1 ? parseInteger(argv[1], "cameras", 4) : settings.cameraCount;
        settings.trackCount = argc > 2 ? parseInteger(argv[2], "tracks", 1) : settings.trackCount;
        settings.viewsPerTrack = argc > 3 ? parseInteger(argv[3], "views", 2) : settings.viewsPerTrack;
        settings.iterations = argc > 4 ? parseInteger(argv[4], "iterations", 1) : settings.iterations;
        settings.threadCount = argc > 5 ? parseInteger(argv[5], "threads", 0) : settings.threadCount;
        settings.repetitions = argc > 6 ? parseInteger(argv[6], "repetitions", 1) : settings.repetitions;
        settings.backends = argc > 7 ? argv[7] : settings.backends;
        settings.deviceIndex = argc > 8 ? parseInteger(argv[8], "device", 0) : settings.deviceIndex;
        settings.mixedPrecision = argc > 9 ? parseBoolean(argv[9], "mixed_precision") : settings.mixedPrecision;
        if (settings.viewsPerTrack > settings.cameraCount)
        {
            throw std::invalid_argument("views must not exceed cameras so each observation uses a unique camera");
        }

        const plabundle::Problem problem = makeProblem(settings);
        const plabundle::ProblemStats stats = plabundle::summarizeProblem(problem);
        std::cout << std::setprecision(12) << "dataset,cameras=" << stats.cameraCount << ",tracks=" << stats.trackCount
                  << ",observations=" << stats.observationCount << ",iterations=" << settings.iterations
                  << ",threads=" << settings.threadCount << ",repetitions=" << settings.repetitions
                  << ",device=" << settings.deviceIndex
                  << ",mixed_precision=" << (settings.mixedPrecision ? "true" : "false") << '\n';

        int failures = 0;
        for (const std::string& backend_name : splitBackends(settings.backends))
        {
            failures += runBackend(problem, settings, parseBackend(backend_name));
        }
        return failures == 0 ? 0 : 2;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "plabundle_benchmark: " << exception.what() << '\n';
        return 2;
    }
}
