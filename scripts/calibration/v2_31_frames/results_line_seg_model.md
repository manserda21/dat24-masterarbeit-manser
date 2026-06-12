Validating /data/manser/runs/pitch_calibration/pitch_lines_v2/weights/best.pt...
Ultralytics 8.4.35 🚀 Python-3.11.15 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 3080, 10002MiB)
YOLO11s-seg summary (fused): 114 layers, 10,073,395 parameters, 0 gradients, 32.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 5.7it/s 0.2s
                   all          6         37      0.674      0.817      0.865      0.758      0.674      0.817       0.86      0.599
         Side line top          6          6      0.878      0.667      0.833      0.463      0.878      0.667      0.764      0.342
      Side line bottom          2          2      0.863          1      0.995      0.896      0.863          1      0.995      0.647
        Side line left          4          4      0.686      0.557      0.694      0.438      0.686      0.557      0.694       0.35
       Side line right          1          1      0.276          1      0.995      0.796      0.276          1      0.995      0.497
           Middle line          3          3      0.539      0.411       0.56      0.408      0.539      0.411       0.56      0.261
    Big rect. left top          4          4          1      0.699      0.888      0.675          1      0.699      0.888       0.62
 Big rect. left bottom          2          2       0.69        0.5      0.502      0.502       0.69        0.5      0.502      0.402
   Big rect. left main          4          4      0.714       0.75      0.724      0.644      0.714       0.75      0.724      0.443
  Small rect. left top          2          3          1      0.493       0.68      0.544          1      0.493       0.68      0.544
 Small rect. left main          2          2      0.899          1      0.995      0.895      0.899          1      0.995      0.796
   Big rect. right top          1          1      0.336          1      0.995      0.895      0.336          1      0.995      0.796
Big rect. right bottom          1          1      0.952          1      0.995      0.995      0.952          1      0.995      0.796
  Big rect. right main          1          1      0.448          1      0.995      0.995      0.448          1      0.995      0.697
 Small rect. right top          1          1      0.511          1      0.995      0.995      0.511          1      0.995      0.796
Small rect. right bottom          1          1      0.565          1      0.995      0.995      0.565          1      0.995      0.895
Small rect. right main          1          1      0.424          1      0.995      0.995      0.424          1      0.995      0.697
Speed: 0.5ms preprocess, 5.0ms inference, 0.0ms loss, 2.6ms postprocess per image
Results saved to /data/manser/runs/pitch_calibration/pitch_lines_v2
💡 Learn more at https://docs.ultralytics.com/modes/train
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode