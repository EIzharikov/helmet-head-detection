from ultralytics import YOLO
from pathlib import Path
import kagglehub

def main():
    path = Path(kagglehub.dataset_download("vodan37/yolo-helmethead"))
    yaml_path = path / "helm" / "helm" / "helm.yaml"

    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        cache=False
    )

if __name__ == '__main__':
    main()