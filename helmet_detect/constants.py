from pathlib import Path

# =====================
# Dataset
# =====================
KAGGLE_DATASET = "vodan37/yolo-helmethead"
DATA_YAML_RELATIVE = Path("helm/helm/helm.yaml")

# =====================
# Experiment
# =====================
PROJECT_NAME = "helmet-head-detection"
DEVICE = 0
PROJECT_ROOT = Path(__file__).parent.parent
