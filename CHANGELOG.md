# Changelog

## 0.1.0 - Unreleased

- Established the package boundary from PlaScan's production bundle-adjustment
  module while preserving its MIT license and numerical conventions.
- Added the independently buildable and installable `plabundle::plabundle`
  CMake package.
- Added pure numeric frame-pinhole/Brown-Conrady camera, problem, option,
  result, backend, and status contracts.
- Added complete Metashape Frame calibration for
  `f/cx/cy/b1/b2/k1..k4/p1..p4`, including exact projection, analytic
  Jacobians, per-parameter masks, grouped bounds/priors, accepted-state
  writeback, and result diagnostics while preserving the historical first
  nine parameter indices and OpenCV Brown convention.
- Added grouped `SolveOptions` for solver, calibration, constraints, backend,
  and quality policy, with lossless conversion to the source-compatible flat
  `Options` layout.
- Added a line-scan numeric problem/solver API with caller-owned projection,
  6DoF image corrections, tie points, fixed/constrained laser ranges, backend
  selection, cancellation, and quality gates; ISD/ISIS parsing stays outside.
- Added an immutable line-scan solve overload whose self-contained result owns
  all refined camera, tie-point, and laser-point state; retained the original
  pointer-based solve as a publish-on-success compatibility adapter.
- Added strict camera/problem validation and projection utilities matching the
  PlaScan extraction baseline.
- Added explicit multi-rig topology with capture poses, shared sensor
  extrinsics/intrinsics, missing-camera observations, analytic rig Jacobians,
  rig-aware Schur connectivity, gauge validation, and accepted-state publication.
- Made rig capture/sensor blocks composable with camera pose priors (including
  correlated covariance and tangent frames), camera-plane constraints, GCPs,
  scale bars, LiDAR planes, and laser ranges, with finite-difference and
  CPU/CUDA/Vulkan/OpenCL parity coverage.
- Added the PlaMatrix CPU reference solver, including analytic Jacobians,
  grouped Brown calibration, normal-equation assembly, general and online
  point-Schur paths, and point back-substitution.
- Added control-point, LiDAR point-to-plane, scale-bar, pose, camera-plane, and
  laser-range constraints with structured pre/post quality diagnostics.
- Added source-compatible 3x3 covariance and square-root-information support
  for control points, with strict validation, exact residual/Jacobian
  whitening, reference-online Schur support, and raw metric RMS reporting.
- Added position-only, rotation-only, and joint 6DoF pose priors with correlated
  covariance or square-root information, explicit tangent frames, strict
  finite/symmetry/positive-definiteness checks, and stable Cholesky whitening.
- Added a project-format-neutral photo-alignment binding with stable camera and
  track identifiers, solver-neutral outcome snapshots, and structured
  differential reports for RMS, camera/point state, valid-track masks,
  termination, and backend selection.
- Added problem-wide point filtering, Auto quality gating, cooperative
  cancellation without partial result publication, and deterministic
  multi-threaded post-processing.
- Added adaptive camera-model assessment and mask/restoration helpers.
- Added CUDA and OpenCL reduced-Schur backends with device assembly,
  block-Jacobi PCG, workspace reuse, guarded mixed precision, and detailed
  device/linear-solver timing diagnostics.
- Reused objective-evaluation partitions, reference-Schur per-thread point
  scratch, back-substitution buffers, and solver step capacity across nonlinear
  iterations to avoid repeated host allocations while preserving deterministic
  reduction order.
- Treats an invalid projection from an Armijo derivative/trial state as an
  inadmissible candidate and continues backtracking or damping, while retaining
  hard failures for the accepted state and preserving solver diagnostics when
  a quality gate rejects a result.
- Added Frame image `LeastSquares`, `Huber`, and `Cauchy` robust losses with a
  validated pixel scale, lossless structured/compatibility option conversion,
  and identical cost/IRLS weighting across fixed/shared calibration and general
  or reference-online Schur paths. Least squares remains the default.
- Added versioned Auto selection thresholds, CUDA-before-OpenCL priority,
  quality-gated CPU reruns, and explicit unavailable/unusable backend fallback.
- Applied the quality gate to explicit CPU/device requests and to every CPU
  fallback, preventing a reprojection- or constraint-degrading rerun from being
  reported usable or published. Constraint growth now uses effective
  measurement uncertainty as its physical zero-residual floor.
- Added strict device-index diagnostics and an FP64 availability gate for
  OpenCL double-precision BA.
- Expanded cross-backend tests to cover joint pose, full Brown calibration,
  all survey constraints, Auto routing, quality fallback, and CPU-only builds.
- Reworked the deterministic benchmark to exercise joint BA and report
  cold/warm backend, device, Schur, workspace, and phase timing metrics.
- Verified solver-capable installed-package consumers for CPU-, CUDA-, and
  OpenCL-enabled PlaMatrix packages.
- Added a PlaMatrix source-boundary test to keep legacy matrix/device types
  out of PlaBundle and require review of new numeric dependencies.
