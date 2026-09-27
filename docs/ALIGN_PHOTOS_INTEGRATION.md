# Align Photos integration map

This document records the numeric boundary between the `metalign` Align Photos
model and PlaBundle. PlaBundle does not include or depend on `metalign`; the
application adapter owns identity lookup, coordinate conversion, and accepted
result publication.

## Camera and calibration

| Align Photos field | PlaBundle field | Required conversion |
| --- | --- | --- |
| `RegisteredCamera::pose.rotation` | `FrameCamera::cameraToWorldRotation` | transpose the world-to-camera matrix |
| `camera_center(RegisteredCamera::pose)` | `FrameCamera::cameraCenter` | copy after conversion to the active metric solver frame |
| `CameraModel::f/cx/cy/b1/b2/k1..k4/p1..p4` | `MetashapeFrameCalibration` | exact field-for-field mapping through `applyMetashapeFrameCalibration` |
| `RegisteredCamera::sensor_id` | `Problem::cameraCalibrationGroupIds` | assign one dense group id per sensor id |
| keypoint `x/y/scale` | `Observation::u/v/measurementScale` | use the remapped active-camera index; keep pixel coordinates unchanged |
| `--robust-loss least-squares` | `SolverControlOptions::imageRobustLoss = ImageRobustLoss::LeastSquares` | default; `imageRobustLossScalePixels` remains positive but is not used by the least-squares formula |
| `--robust-loss huber` | `SolverControlOptions::imageRobustLoss = ImageRobustLoss::Huber` | apply one Huber loss to each whitened two-dimensional image residual block |
| `--robust-loss cauchy` | `SolverControlOptions::imageRobustLoss = ImageRobustLoss::Cauchy` | apply one Cauchy loss to each whitened two-dimensional image residual block |
| `--robust-loss-scale` | `SolverControlOptions::imageRobustLossScalePixels` | copy the finite positive pixel scale exactly; reject zero, negative, NaN, or infinity |

The adapter must use `BrownTangentialConvention::Metashape`, which the
Metashape calibration helper selects automatically. Result publication uses
the inverse pose conversion and writes the accepted shared calibration back to
the owning `SensorState`/camera caches.

The flat compatibility `Options` uses the same field names,
`imageRobustLoss` and `imageRobustLossScalePixels`. Robustification is applied
after `Observation::weight` and `measurementScale` whitening and is shared by
fixed-intrinsic, shared-intrinsic, general Schur, and reference-online Schur
paths.

## Rig topology

| Align Photos field | PlaBundle field | Required conversion |
| --- | --- | --- |
| root/master sensor family | `RigCapture::rigId`, `RigSensor::rigId` | assign one stable integer per independent rig |
| `CameraAcquisitionState::capture_group_id` or root image id | `RigCapture::captureId` | assign one capture id within its rig |
| root camera pose | `RigCapture::rigToWorldRotation`, `rigCenterInWorld` | use the root camera-to-world pose |
| `RegisteredCamera::sensor_id` | `RigSensor::sensorId` | preserve as a stable id or remap densely within the rig |
| `SensorMountState::rotation` | `RigSensor::cameraToRigRotation` | copy; this matches `compose_rig_pose` after the pose transpose above |
| `SensorMountState::translation` | `RigSensor::cameraCenterInRig` | copy in rig-frame metric units |
| `fixed_rotation && fixed_translation` | `RigSensor::fixedExtrinsic` | both true means fixed; both false means one free 6DoF block |
| image/root/sensor relation | `RigCameraBinding` | create exactly one binding for every existing active image |

Missing planes are supported: do not synthesize an image or a binding for an
absent sensor/capture pair. Multiple rigs may coexist in one problem. Each rig
must have at least one fixed sensor extrinsic to define its body frame.

Align Photos treats references on rig slaves as station metadata and optimizes
the root/station reference. For numerical parity, map the reference once to the
root composed camera rather than duplicating it on every slave.

## References, markers, scale, and range

| Align Photos field | PlaBundle field | Required conversion |
| --- | --- | --- |
| `FeatureSet::location_reference` | position rows of `CameraPosePrior` | location in metres; covariance in square metres |
| `FeatureSet::rotation_reference` | rotation rows of `CameraPosePrior` | convert camera-to-reference rotation to camera-to-world in the solver frame; convert degree-squared covariance to radians squared and transform yaw/pitch/roll covariance into the chosen tangent basis |
| joint camera reference | 6x6 `CameraPosePrior::uncertaintyMatrix` | order is `[rotation radians, position metres]`; use `Covariance` or `SqrtInformation` explicitly |
| `MarkerProjectionState` | marker `Track::observations` | resolve image and point indices; use projection accuracy in observation weighting/scale |
| `SparsePoint::location_reference` or `MarkerState::reference` | `Track::controlPointConstraints` | target point plus `IsotropicSigma`, 3x3 `Covariance`, or 3x3 `SqrtInformation` uncertainty |
| marker-marker `ScalebarState` | `ScaleBarConstraint` | resolve both marker points to PlaBundle track indices |
| application laser shot | `LaserRangeConstraint` | camera index, lever arm in camera coordinates, range/sigma, measured image observations, and point mode |

All reference positions, marker coordinates, camera centers, scale-bar lengths,
and ranges must be materialized into one metric solver frame before creating
the problem. PlaBundle deliberately does not apply CRS or target-frame Sim(3)
transforms.

For a marker/GCP with a full covariance, set
`ControlPointConstraint::uncertainty` to
`ControlPointUncertainty::Covariance` and copy the row-major 3x3 covariance in
solver-frame m^2 to `uncertaintyMatrix`. If the adapter already computes a
whitening factor `W` such that the information matrix is `W^T W`, select
`SqrtInformation` and store `W` in `uncertaintyMatrix` in m^-1. Use the legacy
`sigmaMeters` only with `IsotropicSigma`. PlaBundle rejects non-finite,
asymmetric, non-positive-definite covariance and rank-deficient square-root
information instead of silently collapsing it to one scalar.

## Gauge and result publication

Use `GaugePolicy::RequireExplicitGauge` when the adapter can prove the gauge.
Valid anchors include fixed capture geometry, at least three non-collinear
control points, a complete pose-prior configuration with metric baseline, or
the supported combinations checked by `validateProblem`. Use `AutoAnchor` only
when reproducing the adapter's automatic anchor policy is acceptable.

Publish only when `Result::usable()` is true. Apply `refinedRig` to capture and
sensor ownership first, use `refinedCameras` for the composed per-image poses,
then publish valid refined points and shared calibrations. The input problem is
immutable and a cancelled or rejected solve does not contain a partial state to
commit.

## Combinations that still need adapter handling

- A `SensorMountState` with only one of `fixed_rotation` and
  `fixed_translation` set cannot be represented exactly by the current single
  `RigSensor::fixedExtrinsic` flag. Reject it or transform it to a supported
  parameterization; do not free or fix all 6DoF silently.
- Camera-camera and camera-marker scale bars do not map to the current
  point-point `ScaleBarConstraint`. Marker-marker scale bars are supported.
- The replica rolling-shutter motion uses a normalized scan convention that is
  not the same as PlaBundle's current row-time model. The adapter must continue
  to reject non-identity replica rolling-shutter motion until that conversion
  is defined and tested.
- Keyframe roles and disconnected components require an application policy;
  they must not be folded into one connected BA implicitly.
- Align Photos currently has no persistent laser-shot record in
  `ReconstructionResult`; laser ranges can be supplied only by a caller that
  owns equivalent shot metadata.
