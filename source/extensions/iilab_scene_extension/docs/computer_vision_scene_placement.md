# Computer Vision Scene Placement Requirements

## Goal

Build a computer vision system that receives one fixed-camera image of the table and predicts which IILAB pieces are present, where they are, and how they are oriented. The vision model must remain independent from the Isaac Sim extension. The extension should only consume the model output and translate it into USD object placement.

The desired compact output is conceptually:

```text
[]
[(Axis, 0, 1, 2, 90, 180, 0)]
[(Axis, 0.42, 0.10, -0.065, 0, 0, 35), (TopBearing, 0.55, 0.18, -0.065, 0, 0, -12)]
```

For machine use, prefer a structured JSON form:

```json
[
  {
    "object_type": "Axis",
    "position": [0.42, 0.10, -0.065],
    "rotation_euler_deg": [0.0, 0.0, 35.0],
    "confidence": 0.94
  }
]
```

## Separation Of Responsibilities

The vision model should be a standalone package, service, or CLI. It should not import `omni`, Isaac Sim, USD, or the extension package.

The extension should own only:

- Parsing and validating the model output.
- Mapping object type names to extension object keys, for example `Axis` to `axis`.
- Converting the predicted pose into USD/Isaac coordinates.
- Creating, moving, or removing scene objects.
- Reporting invalid or low-confidence predictions to the user.

The vision package should own:

- Image loading and preprocessing.
- Neural network inference.
- Camera calibration and image-to-table coordinate conversion.
- Object detection, instance separation, and pose estimation.
- Confidence scoring.
- Exporting a stable output contract.

## Output Contract

Use table/world metric coordinates, not pixel coordinates, as the model's final output. The extension should not need to understand camera intrinsics to place objects.

Required fields per detected piece:

- `object_type`: one of `Axis`, `BottomHousing`, `TopBearing`.
- `position`: `[x, y, z]` in meters, in the same table/world frame used by the simulation.
- `rotation_euler_deg`: `[roll, pitch, yaw]` in degrees, using a documented convention.
- `confidence`: model confidence from `0.0` to `1.0`.

Recommended optional fields:

- `instance_id`: stable id within one prediction, useful for debugging.
- `rotation_quat_wxyz`: `[w, x, y, z]` if the model can produce quaternions directly.
- `bbox_px`: `[x_min, y_min, x_max, y_max]` for visualization.
- `mask`: encoded segmentation mask path or RLE, if available.
- `pose_confidence`: separate confidence for pose quality.

Recommended canonical JSON:

```json
{
  "schema_version": 1,
  "camera_id": "fixed_table_camera_01",
  "coordinate_frame": "iilab_table_world",
  "units": "meters_degrees",
  "objects": [
    {
      "object_type": "Axis",
      "position": [0.42, 0.10, -0.065],
      "rotation_euler_deg": [0.0, 0.0, 35.0],
      "confidence": 0.94
    }
  ]
}
```

Empty table:

```json
{
  "schema_version": 1,
  "camera_id": "fixed_table_camera_01",
  "coordinate_frame": "iilab_table_world",
  "units": "meters_degrees",
  "objects": []
}
```

## Coordinate And Rotation Requirements

Define these before collecting data:

- Camera intrinsics and distortion parameters.
- Camera-to-table or camera-to-world extrinsics.
- Table plane equation.
- Simulation world origin relative to the physical table.
- Position units, preferably meters.
- Euler convention, preferably intrinsic XYZ or another explicitly documented convention.
- Object pose origin, preferably the USD asset root origin.

For extension placement, the adapter should convert:

```text
model object_type -> scene object key
model position -> USD translation
model rotation_euler_deg -> USD quaternion wxyz
```

If the camera is fixed and the table is planar, estimate `x` and `y` with calibrated image-to-table geometry. For `z`, use the known table placement height per object class unless the model is explicitly trained for full 6D pose.

## Recommended Architecture

Use a two-stage architecture first. It is easier to debug than a single end-to-end 6D pose network.

Stage 1: object detection and instance segmentation

- Input: RGB image from the fixed camera.
- Output: class label, mask, bounding box, and confidence for each piece.
- Suitable models: YOLO segmentation, Mask R-CNN, RT-DETR plus segmentation head, or a lightweight DETR-style detector.
- Why: masks make pose estimation more stable than boxes, especially when multiple parts touch or partially overlap.

Stage 2: pose estimation

- Input: image crop, mask, class label, and camera calibration.
- Output: `x, y, z, roll, pitch, yaw`.
- For mostly flat tabletop placement, estimate:
  - `x, y` from the mask centroid or keypoints projected through the table calibration.
  - `yaw` from mask orientation, learned keypoints, or class-specific symmetry-aware pose fitting.
  - `z, roll, pitch` from object defaults or a class-specific pose head.
- For full 6D pose, use CAD/keypoint fitting or a dedicated pose model such as a keypoint network plus PnP, FoundationPose-style approach, or synthetic-data-trained 6D pose estimator.

Stage 3: output normalizer

- Snaps object names to the known taxonomy.
- Converts pose into the agreed coordinate frame.
- Applies confidence thresholds.
- Handles object symmetries.
- Writes the canonical JSON.

Stage 4: extension adapter

- Reads the JSON.
- Validates schema, units, object types, and pose ranges.
- Creates/removes/moves simulation objects.
- Converts Euler rotations to USD quaternions.
- Optionally shows a placement preview before applying.

## Fixed-Camera Augmentation Rules

Because the camera is fixed, do not use augmentations that imply camera movement unless the real camera can move.

Avoid:

- Random image rotation.
- Random perspective warp.
- Random zoom.
- Random crop that changes camera framing.
- Synthetic camera translation.
- Horizontal or vertical flips, unless the physical setup has that symmetry and labels are transformed correctly.

Allowed and recommended:

- Brightness, contrast, exposure, hue, and saturation jitter.
- White balance shifts.
- Sensor noise.
- Mild blur and sharpening.
- JPEG/compression artifacts if relevant.
- Shadows and lighting intensity changes.
- Background/table material color variation if the real setup varies.
- Small object pose variations on the table.
- Physically plausible partial occlusions by other objects or robot parts.

If training from synthetic images, keep the camera fixed and randomize the scene instead: object placement, object yaw, lighting, material roughness, texture variation, and sensor noise.

## Dataset Requirements

Each labeled sample should include:

- Image path.
- Camera id.
- Camera intrinsics/extrinsics version.
- Object class for each instance.
- Segmentation mask or polygon.
- Object pose in the agreed table/world frame.
- Visibility or occlusion percentage if possible.
- Empty-scene examples.

Include negative and edge cases:

- No pieces on the table.
- One piece.
- Multiple pieces.
- Repeated pieces of the same type.
- Pieces close together.
- Partial occlusion.
- Pieces near workspace boundaries.
- Lighting extremes expected in the real setup.

Keep train/validation/test splits separated by capture session or randomized synthetic seed, not just by individual frame. This reduces leakage from near-identical images.

## Metrics

Track detection metrics:

- Per-class precision and recall.
- False positives on empty-table images.
- Instance count accuracy.
- Mask IoU if using segmentation.

Track placement metrics:

- Position error in meters.
- Yaw error in degrees.
- Full rotation error if predicting 6D pose.
- Success rate under extension placement tolerance.
- Symmetry-aware rotation error for symmetric parts.

Suggested initial thresholds:

- Detection confidence: `>= 0.5` for preview, `>= 0.75` for automatic placement.
- Position error target: below `1-2 cm` if the robot must pick reliably.
- Yaw error target: below `5-10 degrees`, depending on gripper tolerance.

## Symmetry And Pose Ambiguity

Some parts may have rotational symmetries. The model and metrics must not punish equivalent orientations.

Document for each object:

- Whether yaw has 180-degree or other symmetry.
- Whether roll/pitch are fixed on the table.
- Whether multiple stable resting poses are possible.
- Which USD orientation corresponds to the model's zero rotation.

The extension adapter should optionally normalize equivalent rotations before applying them.

## Extension Integration Plan

Add an extension-side adapter later, separate from the model:

```text
cv_output_parser.py
  - load_prediction_json(path)
  - validate_prediction(payload)
  - normalize_object_type(name)
  - euler_deg_to_quat_wxyz(rotation)
  - prediction_to_scene_object_specs(payload)
  - apply_prediction_to_scene(scene, payload)
```

The adapter should not run the neural network. It should only consume finished predictions.

Recommended user flow:

1. User captures or selects an image.
2. Independent vision package writes prediction JSON.
3. Extension imports the JSON.
4. Extension previews objects and confidence.
5. User applies placement.
6. Extension creates/removes/moves objects in simulation.

## Failure Handling

The model should return an empty list when no pieces are detected. It should not invent low-confidence pieces to avoid empty output.

The extension should reject:

- Unknown object types.
- Missing fields.
- Non-finite numbers.
- Positions outside the table workspace.
- Rotations outside the documented convention.
- Low-confidence objects when automatic placement is enabled.
- Multiple detections that overlap too much unless allowed.

The extension should report rejected objects with clear messages instead of silently dropping them.

## Recommended First Milestone

Start with the simplest useful version:

- Fixed RGB camera.
- Three object classes: `Axis`, `BottomHousing`, `TopBearing`.
- Single stable tabletop pose per object type.
- Detect object masks.
- Estimate `x, y` from calibrated table projection.
- Estimate `yaw` from mask/keypoints.
- Use class defaults for `z, roll, pitch`.
- Export canonical JSON.
- Add extension importer that places objects from JSON.

Only move to full 6D pose estimation after this version is reliable.

