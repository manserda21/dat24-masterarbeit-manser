## Befehl

### Distance Treshold 15
```bash
python3 src/ball_possession/detect_possessions.py \
--tracking_csv /data/manser/runs/modric_kroos_bev_tracking_v1/tracking_results_filtered.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_15 \
--distance_threshold 15
```

### Distance Treshold 25
```bash
python3 src/ball_possession/detect_possessions.py \
--tracking_csv /data/manser/runs/modric_kroos_bev_tracking_v1/tracking_results_filtered.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_25 \
--distance_threshold 25
```

### Distance Treshold 35
```bash
python3 src/ball_possession/detect_possessions.py \
--tracking_csv /data/manser/runs/modric_kroos_bev_tracking_v1/tracking_results_filtered.csv \
--output_dir /data/manser/runs/modric_kroos_bev_tracking_v1/possessions/distance_35 \
--distance_threshold 35
```