from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from helmet_detect.constants import DATA_YAML_RELATIVE, DEVICE, RESULTS_PATH


def train(args):
    model = YOLO(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RESULTS_PATH}/{Path(args.model).name}_epochs{args.epochs}_imgsz{args.imgsz}_batch{args.batch}_freeze{args.freeze}_augment{args.augment}_{timestamp}"

    if args.augment:
        augment_kwargs = dict(
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )
    else:
        augment_kwargs = dict(
            augment=False
        )

    model.train(
        data=DATA_YAML_RELATIVE,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=DEVICE,
        project="helmet-head-detection",
        name=run_name,
        plots=True,
        seed=42,
        freeze=args.freeze,
        **augment_kwargs
    )
    print(f"Training completed. Results saved in project '{run_name}'")
