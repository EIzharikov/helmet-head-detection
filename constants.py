# constants.py
from pathlib import Path

# =====================
# Dataset
# =====================
KAGGLE_DATASET = "vodan37/yolo-helmethead"

DATA_YAML_RELATIVE = Path("helm/helm/helm.yaml")

# =====================
# Model
# =====================
MODEL_NAME = "yolov8n.pt"

# =====================
# Training hyperparameters
# =====================
EPOCHS = 2
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0

CACHE = False
SEED = 0

# =====================
# Experiment
# =====================
PROJECT_NAME = "helmet-head-detection"
RUN_NAME = "yolov8n_baseline"
