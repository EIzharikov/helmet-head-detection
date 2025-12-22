from pathlib import Path

import kagglehub
import pandas as pd
from ultralytics import YOLO

from helmet_detect.constants import (DATA_YAML_RELATIVE, DEVICE,
                                     KAGGLE_DATASET, PROJECT_ROOT)


def evaluate(args):
    results = []

    weight_files = sorted(args.weights_dir.glob("*.pt"))
    if not weight_files:
        raise ValueError(f"No .pt files found in {args.weights_dir}")

    for weight_path in weight_files:
        print(f"\nEvaluating {weight_path.name}")

        model = YOLO(weight_path)

        metrics = model.val(
            data=PROJECT_ROOT / DATA_YAML_RELATIVE,
            imgsz=args.imgsz,
            device=DEVICE,
            plots=False,
            save=False,
        )

        row = {
            "model": weight_path.name,
            "params_M": round(model.model.info()[1] / 1e6, 2),
            "imgsz": args.imgsz,
            "precision": round(float(metrics.box.p.mean()), 4),
            "recall": round(float(metrics.box.r.mean()), 4),
            "mAP50": round(float(metrics.box.map50.mean()), 4),
            "mAP50-95": round(float(metrics.box.map.mean()), 4),
            "inference_ms": round(float(metrics.speed["inference"]), 3),
        }

        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("mAP50-95", ascending=False)

    df.to_excel(args.output, index=False)
    print(f"\nSaved benchmark table to {args.output}")
