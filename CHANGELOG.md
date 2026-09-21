# Changelog

## 0.1.0 - Unreleased

- Established the package boundary from PlaScan's production bundle-adjustment
  module while preserving its MIT license and numerical conventions.
- Added the independently buildable and installable `plabundle::plabundle`
  CMake package.
- Added pure numeric frame-pinhole/Brown-Conrady camera, problem, option,
  result, backend, and status contracts.
- Added strict camera/problem validation and projection utilities matching the
  PlaScan extraction baseline.
- Added the PlaMatrix CPU reference solver, including analytic Jacobians,
  grouped Brown calibration, normal-equation assembly, general and online
  point-Schur paths, and point back-substitution.
- Added control-point, LiDAR point-to-plane, scale-bar, pose, camera-plane, and
  laser-range constraints with structured pre/post quality diagnostics.
- Added problem-wide point filtering, Auto quality gating, cooperative
  cancellation without partial result publication, and deterministic
  multi-threaded post-processing.
- Added adaptive camera-model assessment and mask/restoration helpers.
- Added CUDA and OpenCL reduced-Schur backends with device assembly,
  block-Jacobi PCG, workspace reuse, guarded mixed precision, and detailed
  device/linear-solver timing diagnostics.
- Added versioned Auto selection thresholds, CUDA-before-OpenCL priority,
  quality-gated CPU reruns, and explicit unavailable/unusable backend fallback.
- Added strict device-index diagnostics and an FP64 availability gate for
  OpenCL double-precision BA.
- Expanded cross-backend tests to cover joint pose, full Brown calibration,
  all survey constraints, Auto routing, quality fallback, and CPU-only builds.
- Reworked the deterministic benchmark to exercise joint BA and report
  cold/warm backend, device, Schur, workspace, and phase timing metrics.
- Verified solver-capable installed-package consumers for CPU-, CUDA-, and
  OpenCL-enabled PlaMatrix packages.
