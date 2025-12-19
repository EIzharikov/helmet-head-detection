import argparse
from helmet_detect.train import train
from helmet_detect.infer import infer
from helmet_detect.constants import MODEL_NAME

def main():
    parser = argparse.ArgumentParser(description="Helmet Detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ====== TRAIN ======
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--model", default=MODEL_NAME, help="YOLO model name")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")

    # ====== INFER ======
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--weights", required=True, help="Path to trained weights")
    infer_parser.add_argument("--source", required=True, help="Image or folder to infer on")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)

if __name__ == "__main__":
    main()
