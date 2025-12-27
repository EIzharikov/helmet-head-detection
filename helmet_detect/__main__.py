import argparse
from pathlib import Path

from helmet_detect.evaluate_models import evaluate
from helmet_detect.infer import infer
from helmet_detect.train import train


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ====== TRAIN ======
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--model", help="YOLO model name")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument(
        "--freeze", type=int, default=0, help="Amount of freezed layers"
    )
    train_parser.add_argument(
        "--augment", action="store_true", help="Enable data augmentation"
    )

    # ====== INFER ======
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--weights", required=True, help="Path to trained weights"
    )
    infer_parser.add_argument(
        "--source", required=True, help="Image or folder to infer on"
    )

    # ====== EVALUATE ======
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluate")
    eval_parser.add_argument(
        "--weights-dir", type=Path, required=True, help="Path to trained weights"
    )
    eval_parser.add_argument(
        "--output", type=Path, required=True, help="Path to save table"
    )
    eval_parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
