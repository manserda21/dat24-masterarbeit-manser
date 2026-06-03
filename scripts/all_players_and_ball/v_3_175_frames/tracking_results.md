### Befehl:

```bash
cd /home/manser/projects/dat24-masterarbeit-manser

python3 -m src.run_tracking \
    --model /data/manser/runs/all_players_and_ball_v3_yolo11s/weights/best.pt \
    --source /data/manser/test_videos/malaga_modric_clip_10m00s_5min.mp4 \
    --name malaga_modric_tracking \
    --project /data/manser/tracking_outputs \
    --conf 0.35 \
    --iou 0.5
```