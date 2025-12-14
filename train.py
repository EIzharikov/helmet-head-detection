from ultralytics import YOLO
from pathlib import Path
import kagglehub

import constants


def main():
    path = Path(kagglehub.dataset_download("vodan37/yolo-helmethead"))
    yaml_path = path / "helm" / "helm" / "helm.yaml"

    model = YOLO(constants.MODEL_NAME)

    model.train(
        data=yaml_path,
        epochs=constants.EPOCHS,
        imgsz=constants.IMAGE_SIZE,
        batch=constants.BATCH_SIZE,
        device=constants.DEVICE,
        cache=constants.CACHE,
        seed=constants.SEED,
        project=constants.PROJECT_NAME,
        name=constants.RUN_NAME,
    )


if __name__ == "__main__":
    main()
