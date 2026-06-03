
### Aufruf

```bash
python -m src.training.train     --data /data/manser/datasets/all_players_and_ball/all_players_and_ball_dataset_v5/final_dataset_split/data.yaml     --name all_players_and_ball_v5     --model yolo11m.pt     --imgsz 1280     --batch 4     --epochs 150
```

### Ergebnis

150 epochs completed in 0.719 hours.
Optimizer stripped from /data/manser/runs/all_players_and_ball_v5/weights/last.pt, 40.6MB
Optimizer stripped from /data/manser/runs/all_players_and_ball_v5/weights/best.pt, 40.6MB

Validating /data/manser/runs/all_players_and_ball_v5/weights/best.pt...
Ultralytics 8.4.35 🚀 Python-3.11.15 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 3080, 10002MiB)
YOLO11m summary (fused): 126 layers, 20,039,284 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 6.4it/s 1.4s
                   all         70        993      0.926      0.857      0.922      0.769
              opponent         66        467       0.98      0.968      0.991      0.869
                danilo         47         47       0.91      0.787      0.912      0.731
                varane         40         40      0.912       0.95      0.974       0.84
                 ramos         41         42      0.934       0.81      0.875       0.76
               marcelo         46         47      0.901      0.894      0.949      0.808
                modric         56         56      0.951      0.946      0.976      0.838
              casemiro         58         58      0.946      0.914      0.967      0.827
                 kroos         55         55      0.866      0.927      0.941      0.788
                  isco         47         48      0.912      0.833      0.942      0.795
               benzema         37         37      0.893      0.901      0.956      0.771
               ronaldo         41         41      0.971      0.825      0.942      0.808
                  ball         55         55      0.932      0.527      0.644      0.388
Speed: 0.3ms preprocess, 11.4ms inference, 0.0ms loss, 5.7ms postprocess per image
Results saved to /data/manser/runs/all_players_and_ball_v5

==============================
TRAINING FINISHED
==============================