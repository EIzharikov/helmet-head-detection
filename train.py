from ultralytics import YOLO
from pathlib import Path
import kagglehub

import constants
import constants_YOLOv5n


def main():
    path = Path(kagglehub.dataset_download("vodan37/yolo-helmethead"))
    yaml_path = path / "helm" / "helm" / "helm.yaml"

    # Train with YOLOv8n
    # model = YOLO(constants.MODEL_NAME)

    # model.train(
    #     data=yaml_path,
    #     epochs=constants.EPOCHS,
    #     imgsz=constants.IMAGE_SIZE,
    #     batch=constants.BATCH_SIZE,
    #     device=constants.DEVICE,
    #     cache=constants.CACHE,
    #     seed=constants.SEED,
    #     project=constants.PROJECT_NAME,
    #     name=constants.RUN_NAME,
    # )

    # Train with YOLOv5n
    model = YOLO(constants_YOLOv5n.MODEL_NAME)

    model.train(
        data=yaml_path,
        epochs=constants_YOLOv5n.EPOCHS,
        imgsz=constants_YOLOv5n.IMAGE_SIZE,
        batch=constants_YOLOv5n.BATCH_SIZE,
        device=constants_YOLOv5n.DEVICE,
        cache=constants_YOLOv5n.CACHE,
        seed=constants_YOLOv5n.SEED,
        project=constants_YOLOv5n.PROJECT_NAME,
        name=constants_YOLOv5n.RUN_NAME,
    )


if __name__ == "__main__":
    main()
