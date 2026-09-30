# PlaBundle

PlaBundle is a C++20 bundle and control-network adjustment library powered by
PlaMatrix. Its public API is a standalone numeric boundary: frame-pinhole
cameras, Brown-Conrady intrinsics, tracks, survey constraints, a line-scan
adjustment problem, solver options, and structured diagnostics.

Version 0.1 provides a PlaMatrix CPU reference solver plus optional CUDA and
OpenCL reduced-Schur backends with:

- joint point, camera-pose, rig-capture, sensor-extrinsic, and grouped shared-intrinsic refinement;
- source-compatible Brown calibration plus complete Metashape Frame
  `f/cx/cy/b1/b2/k1..k4/p1..p4` calibration;
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
keeps its own CPU build requirements. CUDA, Vulkan, and OpenCL are enabled on PlaMatrix,
not through duplicate PlaBundle options; for example, add
`-DPLAMATRIX_WITH_CUDA=ON`, `-DPLAMATRIX_WITH_VULKAN=ON`, or `-DPLAMATRIX_WITH_OPENCL=ON` to a source-tree
developer build.

## Camera boundary

PlaCamera is the sole owner of camera geometry, projection, calibration and
rig topology. PlaBundle consumes `placamera::FramePinholeNumericState` values
through `Problem::cameras` and `Problem::sharedIntrinsicReferenceCameras`, and
returns the accepted states through `Result::refinedCameras`. PlaBundle does
not define a second camera or rig implementation: the solver consumes the
installed PlaCamera numeric-state, projection, parameter-layout, pose-update
and topology-composition APIs directly.

## PlaMatrix boundary

PlaBundle owns camera, observation, constraint, and BA solver semantics.
PlaMatrix's public Eigen-style `Matrix` and vector types provide the direct
numeric API. PlaBundle's implementation uses reviewed `plamatrix/internal/`
headers for finite statistics, robust loss, LM strategy, block-Schur equations
and solver, and the optional OpenCL runtime. PlaBundle does not depend on the
legacy `DenseMatrix`/`Vec2`-style API or directly manage device buffers. The
accelerated backends remain behind PlaMatrix's internal Schur interface.

`PlaBundle.PlaMatrixBoundary` checks every library, test, benchmark, and
example C++ source for legacy types and reviews the allowed PlaMatrix header
imports. New numeric dependencies should be added to its allowlist only after
checking that they do not move BA-specific models into PlaMatrix. Public
headers are separately checked by `PlaBundle.PublicDependencyBoundary` and
compiled individually by the `public_header_*` test sources.

## Minimal native frame BA

```cpp
#include <plabundle/solver.h>
#include <placamera/frame_numeric_state.h>

// Build the typed PlaCamera models/numeric states in the application layer.
placamera::FramePinholeNumericState left = makeNativeCamera("left", {-1.0, 0.0, 0.0});
placamera::FramePinholeNumericState right = makeNativeCamera("right", {1.0, 0.0, 0.0});

plabundle::Track track;
track.initialPoint = {0.2, -0.1, 6.0};
track.observations = {
    {0, 700.0, 400.0, 1.0, 1.0},
    {1, 300.0, 400.0, 1.0, 1.0},
};

plabundle::Problem problem;
problem.cameras = {left, right};
problem.tracks = {track};

plabundle::SolveOptions options;
options.calibration.refineCameraPose = false;
const plabundle::Result result = plabundle::Solver().solve(problem, options);
```

When several BA stages run sequentially, keep a caller-owned workspace for
the whole reconstruction:

```cpp
plabundle::Solver solver;
plabundle::SolverWorkspace workspace;
const plabundle::Result first = solver.solve(problem, options, workspace);
// Update problem for the next BA stage.
const plabundle::Result next = solver.solve(problem, options, workspace);
workspace.clear(); // optional: release cached structure before the next job
```

The CPU sparse solver reuses symbolic analysis only when the reduced matrix
pattern matches exactly. Different camera or observation graphs rebuild the
pattern automatically. A workspace is for sequential calls and must not be
shared by concurrent solves; the existing `solve(problem, options)` overload
continues to own a short-lived workspace.

`SolveOptions` keeps independent policy areas separate:

- `solver`: iterations, Frame image robust loss, line search, filtering, threading, progress, and cancellation.
- `calibration`: pose and shared Brown-Conrady parameter refinement, bounds, and priors.
- `constraints`: survey-constraint weights and robust-loss thresholds.
- `backend`: CPU/CUDA/Vulkan/OpenCL selection, device, scale thresholds, mixed precision, and fallback.
- `quality`: result acceptance thresholds.

The original flat `Options` remains source-compatible for callers that pass a
named options object. Use the one-argument `solve(problem)` form for defaults;
an untyped `solve(problem, {})` is intentionally avoided because two options
types cannot be selected unambiguously. `makeSolveOptions` and
`makeCompatibilityOptions` provide explicit, lossless conversion without
storing two writable representations in one object.

Frame observations default to exact least squares. Set
`solver.imageRobustLoss` to `ImageRobustLoss::Huber` or
`ImageRobustLoss::Cauchy` and provide the positive
`solver.imageRobustLossScalePixels` to robustify each whitened two-dimensional
reprojection block. The compatibility `Options` exposes the same two field
names. Observation weights and `measurementScale` are applied before the
robust loss; survey-constraint losses remain independently configured under
`constraints`.

### Complete Metashape Frame calibration

PlaCamera owns the Metashape calibration conversion and parameter layout. The
legacy nine parameter block remains at indices `[fx, log(fy/fx), cx, cy, k1,
k2, k3, p1, p2]`; the compatible extension appends `[b2, k4, p3, p4]`. Metashape `f`
and `b1` are represented losslessly as `fy=f` and `fx=f+b1`.

Applying the calibration selects `BrownTangentialConvention::Metashape`. This
keeps the existing OpenCV Brown behavior unchanged for old callers while using
Metashape's complete tangential field, including the `p3/p4` radial scaling,
for imported Metashape cameras. `supportedCameraParameters(camera)` reports
the exact valid mask for that camera; use the camera overload when working
with the appended parameters.

Enable `refineSharedMetashapeParameters` together with shared focal refinement
to optimize `b2/k4/p3/p4`. `sharedIntrinsicParameterMask` can select individual
fields, and the corresponding `maxShared*` and `shared*PriorSigma` members
control bounds and priors. Cameras in one shared calibration group must use
the same projection model and tangential convention. Accepted values are
written to every camera in the group and are also summarized by
`Result::quality.refinedSharedSkewB2`, `refinedSharedRadialK4`,
`refinedSharedTangentialP3`, and `refinedSharedTangentialP4`.

`CameraPosePrior` supports backward-compatible independent position/rotation
sigmas, a correlated covariance, or a square-root information matrix. Priors
may constrain position only, rotation only, or the joint 6DoF pose. Matrices
are row-major 6x6 in `[rotation radians, position meters]` order and may be
expressed in the world tangent frame or the prior camera frame. Covariances
must be finite, symmetric, and positive definite; square-root information is
validated through its positive-definite information matrix. Whitening is
performed before robust loss evaluation.

All cameras, points, priors, and distances must already use one
caller-defined numeric frame. PlaBundle deliberately does not infer CRS,
units, image identity, or project metadata.

## Photo-alignment adapter and differential checks

`<plabundle/photo_alignment.h>` binds the dense indices in `Problem` to stable,
caller-owned camera and track identifiers. Identifiers are opaque strings, so
an application may use project database keys, image names, or exchange-format
IDs without exposing Metashape, COLMAP, ISIS, Qt, JSON, or file-format types in
PlaBundle's public API.

`makePhotoAlignmentOutcome` converts a `Result` into a solver-neutral snapshot
containing termination status/reason, requested and used backends, RMS values,
refined cameras, refined points, and the valid-track mask. A legacy solver or
another BA library can populate the same `PhotoAlignmentOutcome` contract.
`comparePhotoAlignmentOutcomes` then aligns states by identifier rather than
storage order and reports per-camera center/rotation differences, per-track
point differences, valid-mask mismatches, RMS differences, termination
differences, and backend-selection differences. Numeric thresholds and which
categorical fields are required to match are explicit in
`PhotoAlignmentComparisonTolerance`.

This layer performs no project parsing or implicit coordinate conversion. The
application remains responsible for converting its camera and control-network
records into the numeric `Problem` exactly once.

## Rig BA

`<placamera/camera_topology.h>` separates a rig into capture poses, reusable sensor
extrinsics, and camera bindings. Capture poses use rig-to-world rotations and
rig centers in world coordinates. Sensor extrinsics use camera-to-rig
rotations and camera centers in rig coordinates. The composed camera pose is
`R_camera_world = R_rig_world * R_camera_rig` and
`C_camera_world = C_rig_world + R_rig_world * C_camera_rig`.

Set `Problem::rig` to enable the topology. Every camera model has exactly one
binding, while observations may omit any bound camera. A sensor marked
`fixedExtrinsic = false` owns one shared extrinsic block across all captures;
at least one fixed sensor per rig defines the rig body frame. Capture
`fixedPose` flags and the normal gauge policy control global gauge freedom.
When calibration refinement is enabled and no explicit calibration groups are
provided, cameras bound to the same rig sensor automatically share one
intrinsic/model block. `Result::refinedRig` publishes the accepted capture and
sensor parameters, while `refinedCameras` contains their composed poses.

Pose priors, camera-plane constraints, control points, LiDAR point-to-plane
constraints, scale bars, and laser ranges may be used in the same rig problem.
Pose- and range-derived camera Jacobians are chained to the active capture and
shared sensor-extrinsic blocks; they are not attached to a detached per-image
pose. A missing frame is represented by omitting that capture/sensor pair from
`cameraBindings` and from `cameras`, not by creating a dummy camera:

```cpp
plabundle::Problem problem;
problem.cameras = camera_models;       // existing images only
problem.tracks = tracks;
problem.rig.captures = captures;       // may contain multiple rigId values
problem.rig.sensors = sensors;         // one fixedExtrinsic sensor per rig
problem.rig.cameraBindings = bindings; // one binding per existing image
problem.cameraPosePriors = pose_priors;
problem.scaleBarConstraints = scale_bars;
problem.laserRangeConstraints = laser_shots;
problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;

plabundle::SolveOptions options;
options.calibration.refineCameraPose = true;
const plabundle::Result result = plabundle::Solver().solve(problem, options);
if (result.usable())
{
    publish(result.refinedRig, result.refinedCameras, result.points);
}
```

## Control-point uncertainty

`ControlPointConstraint` keeps `sigmaMeters` as the source-compatible default:
leave `uncertainty` set to `ControlPointUncertainty::IsotropicSigma` for an
isotropic standard deviation. Set it to `Covariance` and provide a row-major
3x3 covariance in `uncertaintyMatrix` (units m^2) for anisotropic or correlated
GCP uncertainty. Callers that already own a whitening factor may select
`SqrtInformation` and provide that row-major 3x3 factor (units m^-1) directly.

Covariance matrices must be finite, symmetric, and positive definite;
square-root information matrices must be finite and full rank. The solver
whitens both the three-dimensional residual and its point Jacobian, and uses
the same information in normal, shared-intrinsic, and reference-online Schur
paths. `validateControlPointConstraint` performs the public preflight check.
Constraint quality remains an unwhitened Euclidean RMS in metres, so reports
remain directly comparable across uncertainty representations.

See [the Align Photos integration map](docs/ALIGN_PHOTOS_INTEGRATION.md) for
the exact `metalign` field mapping and the combinations that still require an
adapter-side conversion or rejection.

## Line-scan BA

`<plabundle/linescan.h>` provides `plabundle::linescan::Problem`, `Options`,
`Result`, and `solve`. The caller supplies a projection callback returning the
two raw pixel residuals at each observed line. PlaBundle owns the 6DoF
per-image correction, tie-point and optional fixed/constrained laser-point
blocks, robust Schur/LM solve, backend selection, cancellation, and quality
gate. Camera trajectory evaluation, pixel conventions, control-network
parsing, and time/frame conversion remain caller responsibilities. The
callback and any camera objects it captures must stay alive during `solve`;
parallel solves require a thread-safe callback.

The primary line-scan entry point keeps the input immutable and publishes an
accepted solution through the result:

```cpp
const plabundle::linescan::Result result =
    plabundle::linescan::solve(problem, options);
if (result.usable())
{
    useCameraCorrections(result.refinedCameraParameters);
    useTiePoints(result.refinedTiePoints);
    useLaserPoints(result.refinedLaserPoints);
}
else
{
    reportError(result.message);
}
```

The pointer-based
`bool solve(Problem*, Options, Result*, std::string*)` overload is retained as
a compatibility adapter. It writes refined state back to `Problem` only after
the same immutable solve has produced an accepted result.

## Backends

`PlaMatrixCpu` is always available. `PlaMatrixCuda`, `PlaMatrixVulkan`, and `PlaMatrixOpenCl` are
available when the linked PlaMatrix package contains that backend and the
requested runtime device is usable. All four backends run the same nonlinear
driver; accelerated backends move reduced-Schur assembly and block-Jacobi PCG
to the selected device.

`Auto` keeps point-only and small joint problems on the CPU. For joint BA it
prefers CUDA at 128 cameras and 30,000 observations, then Vulkan and OpenCL at 160 cameras
and 50,000 observations. Dense-observation thresholds provide a second route
for unusually dense camera networks. A selected accelerated candidate is
checked by the same quality gate and is rerun on the CPU when the candidate is
unusable or rejected. The CPU rerun must independently pass the gate before it
can be published. `Result::requestedBackend`, `usedBackend`,
`backendFallback`, `backendSelectionReason`, and `qualityGateRejected` make the
decision observable.

Explicit accelerated requests are fail-closed when
`Options::allowBackendFallback` is false. When fallback is enabled, an
unavailable, unusable, or quality-rejected explicit device backend is rerun on
`PlaMatrixCpu`. Explicit CPU and device results use the same gate as `Auto`, and
a degraded CPU fallback is returned as unusable rather than applied. Constraint
growth uses the effective measurement uncertainty as a physical noise floor,
so moving an initially exact GCP or scale bar within its stated uncertainty is
not rejected solely because its initial residual was zero.

CUDA uses `Options::plaMatrixDevice` directly. PlaMatrix's OpenCL runtime
selects one process-wide device at first use; select it before process startup
with `PLAMATRIX_OPENCL_DEVICE_INDEX` or `PLAMATRIX_OPENCL_DEVICE`, then set
`Options::plaMatrixDevice` to the same stable index. PlaBundle uses double
precision and rejects OpenCL devices without FP64 support before dispatch.
Vulkan currently accepts device index `0` and uses that compute device for FP32 block-PCG while keeping the public BA
problem and result in double precision; this mixed-precision path is enabled automatically when Vulkan is selected.

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
  plamatrix_cpu,plamatrix_cuda,plamatrix_vulkan,plamatrix_opencl,auto \
  0 0
```

The first repetition for each backend is labeled `cold`; later repetitions are
`warm`. Output separates setup, assembly, linear solve, back-substitution,
solver total, and API wall time, and reports the actual backend, device,
fallback, Schur pattern reuse, and device-assembly status.

The nonlinear driver keeps normal-equation partitions, objective-evaluation
scratch, reference-Schur point-elimination/back-substitution buffers, reduced
CSR topology, sparse symbolic analysis, and backend resident state alive for
the full solve. Rejected damping retries update numeric values without
discarding reusable topology or vector capacity.

Shared-intrinsic solves build each iteration's effective camera once per
camera and reuse it for observation linearization and cost-only Armijo
evaluation. Reference online Schur assembly normally gives each OpenMP worker
an exclusive off-diagonal accumulation slot, avoiding atomic updates on the
hot path. The aggregate slot storage is capped at 128 MiB; larger dense camera
graphs share slots and use atomic accumulation instead of multiplying memory
by the worker count.

## Consume

```cmake
find_package(plabundle CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE plabundle::plabundle)
```

The installed package resolves PlaMatrix with `find_dependency`. Put both
installation prefixes in `CMAKE_PREFIX_PATH` when they are installed
separately.

## Current limits

- ISD/ISIS/PVL parsing remains outside PlaBundle; callers may use the formal
  trajectory/timing line-scan model or the compatibility projection callback.
- A rig sensor extrinsic is fixed or free as one 6DoF block; independently
  fixing only its rotation or translation requires an adapter-side parameter
  transform and is not represented by `RigSensor`.
- `ScaleBarConstraint` currently connects two track/point blocks. Camera-to-camera
  and camera-to-marker scale bars require adapter-side conversion or rejection.
- OpenCL device selection is process-wide because it is owned by the PlaMatrix
  runtime; a process cannot switch OpenCL devices after first use.
- The v0.x C++ ABI is not yet stable.
- Public headers contain no PlaScan, Qt, OpenCV, GDAL, or PlaPoint types.
