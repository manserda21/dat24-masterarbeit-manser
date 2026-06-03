EarlyStopping: Training stopped early as no improvement observed in last 30 epochs. Best results observed at epoch 64, best model saved as best.pt.
To update EarlyStopping(patience=30) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

94 epochs completed in 0.198 hours.
Optimizer stripped from /data/manser/runs/all_players_and_ball_v4_no_referee_yolo11s/weights/last.pt, 19.3MB
Optimizer stripped from /data/manser/runs/all_players_and_ball_v4_no_referee_yolo11s/weights/best.pt, 19.3MB

Validating /data/manser/runs/all_players_and_ball_v4_no_referee_yolo11s/weights/best.pt...
Ultralytics 8.4.35 🚀 Python-3.11.15 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 3080, 10002MiB)
YOLO11s summary (fused): 101 layers, 9,417,444 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 5.7it/s 1.2s
                   all         49        646      0.871      0.827      0.877      0.708
              opponent         45        310      0.954      0.968      0.987      0.837
                danilo         35         36      0.883      0.889      0.896      0.608
                varane         22         22      0.893       0.76      0.883      0.729
                 ramos         19         20      0.709        0.7      0.753      0.618
               marcelo         28         28      0.903      0.929      0.948      0.817
                modric         33         33      0.927      0.879      0.943      0.773
              casemiro         34         34      0.887      0.923      0.965      0.844
                 kroos         31         31        0.8      0.839      0.902      0.749
                  isco         34         36      0.707      0.833      0.848      0.723
               benzema         29         29      0.877      0.931      0.904      0.722
               ronaldo         32         32      0.917      0.688      0.793      0.661
                  ball         35         35          1      0.588      0.697      0.412
Speed: 0.6ms preprocess, 10.6ms inference, 0.1ms loss, 7.1ms postprocess per image
Results saved to /data/manser/runs/all_players_and_ball_v4_no_referee_yolo11s

==============================
TRAINING FINISHED
==============================