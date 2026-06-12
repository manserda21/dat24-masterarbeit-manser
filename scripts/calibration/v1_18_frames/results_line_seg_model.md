100 epochs completed in 0.046 hours.
Optimizer stripped from /data/manser/runs/pitch_calibration/pitch_lines_v1/weights/last.pt, 20.6MB
Optimizer stripped from /data/manser/runs/pitch_calibration/pitch_lines_v1/weights/best.pt, 20.6MB

Validating /data/manser/runs/pitch_calibration/pitch_lines_v1/weights/best.pt...
Ultralytics 8.4.35 🚀 Python-3.11.15 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 3080, 10002MiB)
YOLO11s-seg summary (fused): 114 layers, 10,073,395 parameters, 0 gradients, 32.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 9.7it/s 0.1s
                   all          3         18      0.583      0.833       0.81      0.557      0.744      0.828      0.817      0.511
         Side line top          3          3      0.644      0.333      0.696      0.268          1          0      0.213      0.168
        Side line left          2          2      0.287        0.5      0.448       0.21      0.792          1      0.995      0.212
           Middle line          1          1      0.613          1      0.995      0.415      0.711          1      0.995      0.796
    Big rect. left top          2          2      0.857          1      0.995      0.846          1      0.786      0.995      0.846
 Big rect. left bottom          2          2       0.73          1      0.995      0.945      0.781          1      0.995      0.746
   Big rect. left main          2          2      0.442          1      0.995      0.497       0.49          1      0.995      0.195
  Small rect. left top          2          3      0.756      0.667      0.672      0.488      0.835      0.667      0.672      0.488
Small rect. left bottom          1          1      0.365          1      0.497      0.497      0.392          1      0.497      0.348
 Small rect. left main          2          2      0.554          1      0.995      0.846      0.693          1      0.995      0.796
Speed: 0.5ms preprocess, 5.0ms inference, 0.0ms loss, 2.8ms postprocess per image
Results saved to /data/manser/runs/pitch_calibration/pitch_lines_v1
💡 Learn more at https://docs.ultralytics.com/modes/train
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode