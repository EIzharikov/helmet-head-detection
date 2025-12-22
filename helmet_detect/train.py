from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from helmet_detect.constants import DATA_YAML_RELATIVE, DEVICE, RESULTS_PATH


def train(args):
    model = YOLO(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RESULTS_PATH}/{Path(args.model).name}_epochs{args.epochs}_imgsz{args.imgsz}_batch{args.batch}_freeze{args.freeze}_{timestamp}"

    model.train(
        data=DATA_YAML_RELATIVE,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=DEVICE,
        project="helmet-head-detection",
        name=run_name,
        save_period=1,
    )
    print(f"Training completed. Results saved in project '{run_name}'")
