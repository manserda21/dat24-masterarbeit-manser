## Pitch calibration v3

```bash
python3 -m src.calibration.pitch_calibration_v3 \
  --dataset_path /data/manser/datasets/calibration/malaga_real_2016_pitch_calibration_dataset_points_lines_v1 \
  --output_dir /data/manser/homography_outputs/malaga_real_calib_v3
```

### Outputs

- `homographies.npz`: same matrix format as v2, usable by `track_and_project.py`
- `homography_quality.csv`: frame-level quality summary
- `homography_point_errors.csv`: error for every point or line-derived intersection
- `debug_overlays/`: original image with annotation/projection differences
- `bev/`: warped bird's-eye-view images

### Point-only fallback

Use this if line labels are not recognized or line intersections hurt quality:

```bash
python3 -m src.calibration.pitch_calibration_v3 \
  --dataset_path /data/manser/datasets/calibration/malaga_real_2016_pitch_calibration_dataset_points_lines_v1 \
  --output_dir /data/manser/homography_outputs/malaga_real_calib_v3_points_only \
  --no_line_intersections
```

### Exclude bad correspondence labels

Use this if `homography_point_errors.csv` shows systematic outliers:

```bash
python3 -m src.calibration.pitch_calibration_v3 \
  --dataset_path /data/manser/datasets/calibration/malaga_real_2016_pitch_calibration_dataset_points_lines_v1 \
  --output_dir /data/manser/homography_outputs/malaga_real_calib_v3_no_center_circle_right \
  --exclude_labels CENTER_CIRCLE_RIGHT
```
