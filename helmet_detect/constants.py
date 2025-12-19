from pathlib import Path

# =====================
# Dataset
# =====================
KAGGLE_DATASET = "vodan37/yolo-helmethead"
DATA_YAML_RELATIVE = Path("helm/helm/helm.yaml")

# полный путь к yaml после скачивания
def get_yaml_path(dataset_path: Path):
    return dataset_path / DATA_YAML_RELATIVE

# =====================
# Experiment
# =====================
PROJECT_NAME = "helmet-head-detection"
RUN_NAME = "yolov8n_baseline"
DEVICE = 0
PROJECT_ROOT = Path(__file__).parent.parent
