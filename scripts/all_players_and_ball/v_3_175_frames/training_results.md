Aufruf:

```bash
cd /home/manser/projects/dat24-masterarbeit-manser

python3 -m src.train \
    --data /data/manser/datasets/all_players_and_ball/all_players_and_ball_v3_175_frames/selected_split/data.yaml \
    --name all_players_and_ball_v3_yolo11s \
    --model yolo11s.pt \
    --imgsz 1280 \
    --batch 2 \
    --epochs 150 \
    --workers 8 \
    --patience 40
```

Validating /data/manser/runs/all_players_and_ball_v3_yolo11s/weights/best.pt...
Ultralytics 8.4.35 🚀 Python-3.11.15 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 3080, 10002MiB)
YOLO11s summary (fused): 101 layers, 9,418,218 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 16.4it/s 0.5s
                   all         35        467      0.713      0.705       0.71      0.554
              opponent         31        206      0.934      0.985      0.989      0.801
                danilo         24         25      0.798       0.76      0.832      0.517
                varane         16         16      0.833      0.875       0.83        0.7
                 ramos         14         15      0.768      0.664      0.628      0.527
               marcelo         21         21      0.687      0.905      0.881      0.697
                modric         25         25       0.75       0.76      0.854      0.653
              casemiro         25         25       0.81       0.92      0.932      0.755
                 kroos         25         25       0.66       0.76      0.766      0.627
                  isco         23         25      0.639       0.76      0.691      0.537
               benzema         21         21      0.505      0.857      0.838      0.682
               ronaldo         22         22      0.321      0.727      0.693      0.576
               referee         13         16       0.33      0.308       0.32      0.227
           person_temp          1          1          1          0          0          0
                  ball         24         24      0.952      0.583       0.68      0.451
Speed: 0.4ms preprocess, 8.8ms inference, 0.2ms loss, 3.5ms postprocess per image
Results saved to /data/manser/runs/all_players_and_ball_v3_yolo11s

==============================
TRAINING FINISHED
==============================