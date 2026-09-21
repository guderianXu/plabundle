# PlaBundle

PlaBundle is a C++20 bundle and control-network adjustment library powered by
PlaMatrix. Its public API is a standalone numeric boundary: frame-pinhole
cameras, Brown-Conrady intrinsics, tracks, survey constraints, solver options,
and structured diagnostics.

Version 0.1 provides a PlaMatrix CPU reference solver plus optional CUDA and
OpenCL reduced-Schur backends with:

- joint point, camera-pose, and grouped shared-intrinsic refinement;
- full Brown `f/aspect/cx/cy/k1/k2/k3/p1/p2` calibration;
- fixed blocks, explicit or automatic gauge handling, and online point Schur;
- control-point, LiDAR point-to-plane, scale-bar, pose, camera-plane, and laser-range constraints;
- robust filtering, quality gates, cancellation without partial publication,
  and adaptive camera-model assessment.
- device-side Schur assembly, block-Jacobi PCG, reusable workspaces, guarded
  mixed precision, automatic backend selection, and observable CPU fallback.

## Build

Against an installed PlaMatrix package:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPLABUNDLE_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/plamatrix/install
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /path/to/plabundle/install
```

For a source-tree developer build, point PlaBundle at a PlaMatrix checkout:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPLABUNDLE_PLAMATRIX_SOURCE_DIR=/path/to/plamatrix \
  -DPLABUNDLE_BUILD_TESTS=ON
```

Set `PLABUNDLE_ENABLE_OPENMP=OFF` to disable PlaBundle's OpenMP loops. PlaMatrix
keeps its own CPU build requirements. CUDA and OpenCL are enabled on PlaMatrix,
not through duplicate PlaBundle options; for example, add
`-DPLAMATRIX_WITH_CUDA=ON` or `-DPLAMATRIX_WITH_OPENCL=ON` to a source-tree
developer build.

## Minimal frame BA

```cpp
#include <plabundle/solver.h>

plabundle::FrameCamera left;
left.cameraCenter = {-1.0, 0.0, 0.0};
left.focalXPixels = left.focalYPixels = 1000.0;
left.principalXPixel = 500.0;
left.principalYPixel = 400.0;

plabundle::FrameCamera right = left;
right.cameraCenter = {1.0, 0.0, 0.0};

plabundle::Track track;
track.initialPoint = {0.2, -0.1, 6.0};
track.observations = {
    {0, 700.0, 400.0, 1.0, 1.0},
    {1, 300.0, 400.0, 1.0, 1.0},
};

plabundle::Problem problem;
problem.cameras = {left, right};
problem.tracks = {track};

plabundle::Options options;
options.refineCameraPose = false;
const plabundle::Result result = plabundle::Solver().solve(problem, options);
```

All cameras, points, priors, and distances must already use one
caller-defined numeric frame. PlaBundle deliberately does not infer CRS,
units, image identity, or project metadata.

## Backends

`PlaMatrixCpu` is always available. `PlaMatrixCuda` and `PlaMatrixOpenCl` are
available when the linked PlaMatrix package contains that backend and the
requested runtime device is usable. All three backends run the same nonlinear
driver; accelerated backends move reduced-Schur assembly and block-Jacobi PCG
to the selected device.

`Auto` keeps point-only and small joint problems on the CPU. For joint BA it
prefers CUDA at 128 cameras and 30,000 observations, then OpenCL at 160 cameras
and 50,000 observations. Dense-observation thresholds provide a second route
for unusually dense camera networks. A selected accelerated candidate is
checked by the same quality gate and is rerun on the CPU when the candidate is
unusable or rejected. `Result::requestedBackend`, `usedBackend`,
`backendFallback`, `backendSelectionReason`, and `qualityGateRejected` make the
decision observable.

Explicit accelerated requests are fail-closed when
`Options::allowBackendFallback` is false. When fallback is enabled, an
unavailable or unusable explicit device backend is rerun on `PlaMatrixCpu`.

CUDA uses `Options::plaMatrixDevice` directly. PlaMatrix's OpenCL runtime
selects one process-wide device at first use; select it before process startup
with `PLAMATRIX_OPENCL_DEVICE_INDEX` or `PLAMATRIX_OPENCL_DEVICE`, then set
`Options::plaMatrixDevice` to the same stable index. PlaBundle uses double
precision and rejects OpenCL devices without FP64 support before dispatch.

`Options::enablePlaMatrixMixedPrecision` enables a guarded FP32 PCG seed for
accelerated double-precision solves. `Result::plaMatrix.mixedPrecisionUsed` is
true only when that seed converged and was accepted; `false` can mean the solve
safely continued in FP64.

Build the deterministic joint-pose benchmark with
`-DPLABUNDLE_BUILD_BENCHMARKS=ON`. It accepts camera/track/view counts,
iterations, threads, repetitions, a backend list, device index, and a mixed
precision switch:

```bash
./build/plabundle_benchmark \
  128 4000 8 4 8 3 \
  plamatrix_cpu,plamatrix_cuda,plamatrix_opencl,auto \
  0 0
```

The first repetition for each backend is labeled `cold`; later repetitions are
`warm`. Output separates setup, assembly, linear solve, back-substitution,
solver total, and API wall time, and reports the actual backend, device,
fallback, Schur pattern reuse, and device-assembly status.

## Consume

```cmake
find_package(plabundle CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE plabundle::plabundle)
```

The installed package resolves PlaMatrix with `find_dependency`. Put both
installation prefixes in `CMAKE_PREFIX_PATH` when they are installed
separately.

## Current limits

- Frame-pinhole/Brown cameras only; line-scan and ISIS/PVL integration are not
  part of version 0.1.
- OpenCL device selection is process-wide because it is owned by the PlaMatrix
  runtime; a process cannot switch OpenCL devices after first use.
- The v0.x C++ ABI is not yet stable.
- Public headers contain no PlaScan, Qt, OpenCV, GDAL, or PlaPoint types.
