from pathlib import Path
from ultralytics import YOLO
from helmet_detect.constants import PROJECT_ROOT

def infer(args):
    model = YOLO(args.weights)
    model.predict(source=args.source, save=True)
    print(f"Inference completed. Results saved in '{PROJECT_ROOT}/runs/detect'")
