python3 src/ball_possession/export_possession_frames.py \
--video /data/manser/videos/2nd_half_clip_first5min.mkv \
--possessions_csv /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_25/detected_possessions.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions

### Distance Treshold 15
```bash
python3 src/ball_possession/export_possession_frames.py \
--video /data/manser/videos/2nd_half_clip_first5min.mkv \
--possessions_csv /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_15/detected_possessions.csv \
--tracking_csv /data/manser/runs/modric_kroos_bev_tracking_v1/tracking_results_filtered.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_15
```

### Distance Treshold 25
```bash
python3 src/ball_possession/export_possession_frames.py \
--video /data/manser/videos/2nd_half_clip_first5min.mkv \
--possessions_csv /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_25/detected_possessions.csv \
--tracking_csv /data/manser/runs/modric_kroos_bev_tracking_v1/tracking_results_filtered.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_25
```

### Distance Treshold 35
```bash
python3 src/ball_possession/export_possession_frames.py \
--video /data/manser/videos/2nd_half_clip_first5min.mkv \
--possessions_csv /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_35/detected_possessions.csv \
--tracking_csv /data/manser/runs/modric_kroos_bev_tracking_v1/tracking_results_filtered.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_35
```