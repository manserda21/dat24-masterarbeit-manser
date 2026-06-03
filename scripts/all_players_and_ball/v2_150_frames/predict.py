from ultralytics import YOLO

model = YOLO(
    "/data/manser/runs/all_players_and_ball_v2_yolo11s/weights/best.pt"
)

results = model.predict(

    source="/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames/selected_split/images/val",

    imgsz=1280,
    conf=0.25,

    save=True,
    save_txt=False,
    save_conf=True
)