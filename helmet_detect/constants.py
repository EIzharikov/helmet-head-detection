from pathlib import Path

import kagglehub

# =====================
# Dataset
# =====================
KAGGLE_DATASET_NAME = "vodan37/yolo-helmethead"
KAGGLE_DATASET = Path(kagglehub.dataset_download(KAGGLE_DATASET_NAME))
DATA_YAML_RELATIVE = KAGGLE_DATASET / "helm" / "helm" / "helm.yaml"

# =====================
# Experiment
# =====================
PROJECT_NAME = "helmet-head-detection"
DEVICE = 0
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / "results"
