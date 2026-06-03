from ultralytics import YOLO
import argparse

from config import RUNS_DIR


# =========================================================
# ARGUMENT PARSER
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train YOLO model"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="YOLO model"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience"
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_arguments()

    print("\n==============================")
    print("YOLO TRAINING")
    print("==============================")

    print(f"Data:      {args.data}")
    print(f"Model:     {args.model}")
    print(f"Run name:  {args.name}")

    print(f"\nImage size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs:     {args.epochs}")

    print("\nLoading model...\n")

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = YOLO(args.model)

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train(

        # -------------------------------------------------
        # DATA
        # -------------------------------------------------

        data=args.data,

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,

        # -------------------------------------------------
        # HARDWARE
        # -------------------------------------------------

        device=args.device,
        workers=args.workers,
        
        
        # -------------------------------------------------
        # PERFORMANCE
        # -------------------------------------------------
        
        cache=True,
        amp=True,
        
        # -------------------------------------------------
        # OPTIMIZATION
        # -------------------------------------------------

        optimizer="AdamW",
        patience=args.patience,

        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------

        project=RUNS_DIR,
        name=args.name,

        # -------------------------------------------------
        # AUGMENTATION
        # -------------------------------------------------

        degrees=3.0,
        translate=0.05,
        scale=0.4,

        fliplr=0.5,

        mosaic=1.0,
        mixup=0.0,

        close_mosaic=10,

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val=True,
        plots=True,

        # -------------------------------------------------
        # REPRODUCIBILITY
        # -------------------------------------------------

        seed=42,
        deterministic=True,
    )

    print("\n==============================")
    print("TRAINING FINISHED")
    print("==============================\n")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()