from datetime import datetime
from pathlib import Path
import kagglehub
from ultralytics import YOLO
from helmet_detect.constants import DEVICE, KAGGLE_DATASET, PROJECT_ROOT, get_yaml_path

def train(args):
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    yaml_path = get_yaml_path(dataset_path)

    model = YOLO(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(args.model).stem}_{timestamp}"

    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=DEVICE,
        project="helmet-head-detection",
        name=run_name,
    )
    print(f"Training completed. Results saved in project '{PROJECT_ROOT}/runs/detect/{run_name}'")
