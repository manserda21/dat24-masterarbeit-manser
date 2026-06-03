Im Projektordner

```bash
cd /home/manser/projects/dat24-masterarbeit-manser
```

```bash
PYTHONPATH=. python src/train.py \
--data /data/manser/datasets/modric_kroos_ball_dataset_v1/processed/data.yaml \
--name modric_kroos_ball_v1 \
--model yolov8m.pt \
--imgsz 1280 \
--batch 2 \
--epochs 150
```