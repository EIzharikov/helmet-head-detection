import re
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from helmet_detect.constants import (DATA_YAML_RELATIVE, PROJECT_ROOT,
                                     get_device)

RUN_PATTERN = re.compile(
    r"""
    (?P<model>yolo(v)?\d+\w*)\.pt_
    epochs(?P<epochs>\d+)_
    imgsz(?P<imgsz>\d+)_
    batch(?P<batch>\d+)_
    freeze(?P<freeze>\d+)_
    augment(?P<augment>True|False)_
    (?P<timestamp>\d+_\d+)
    """,
    re.VERBOSE,
)


def parse_run_name(run_dir: Path) -> dict:
    match = RUN_PATTERN.match(run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse run directory name: {run_dir.name}")

    data = match.groupdict()
    data["epochs"] = int(data["epochs"])
    data["imgsz"] = int(data["imgsz"])
    data["batch"] = int(data["batch"])
    data["freeze"] = int(data["freeze"])
    data["augment"] = data["augment"] == "True"

    model_name = data["model"]

    if model_name.startswith("yolov5"):
        data["model_family"] = "YOLOv5"
    elif model_name.startswith("yolov8"):
        data["model_family"] = "YOLOv8"
    elif model_name.startswith("yolo11"):
        data["model_family"] = "YOLO11"
    else:
        data["model_family"] = "Unknown"

    return data


def evaluate(args):
    results = []

    run_dirs = [p for p in args.weights_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise ValueError(f"No run directories found in {args.weights_dir}")

    for run_dir in sorted(run_dirs):
        print(f"\nProcessing run: {run_dir.name}")

        meta = parse_run_name(run_dir)
        weights_dir = run_dir / "weights"

        weight_name = "best.pt"
        weight_path = weights_dir / weight_name
        if not weight_path.exists():
            print(f"Missing {weight_name} in {run_dir.name}, skipping")
            continue

        print(f"  🔍 Evaluating {weight_name}")

        model = YOLO(weight_path)

        metrics = model.val(
            data=PROJECT_ROOT / DATA_YAML_RELATIVE,
            imgsz=meta["imgsz"],
            device=get_device(),
            plots=False,
            save=False,
        )

        row = {
            "run_name": run_dir.name,
            "model_family": meta["model_family"],
            "model": meta["model"],
            "weights_type": weight_name.replace(".pt", ""),
            "epochs": meta["epochs"],
            "imgsz": meta["imgsz"],
            "batch": meta["batch"],
            "freeze": meta["freeze"],
            "augment": meta["augment"],
            "params_M": round(model.model.info()[1] / 1e6, 2),
            "precision": round(float(metrics.box.p.mean()), 4),
            "recall": round(float(metrics.box.r.mean()), 4),
            "mAP50": round(float(metrics.box.map50.mean()), 4),
            "mAP50-95": round(float(metrics.box.map.mean()), 4),
            "inference_ms": round(float(metrics.speed["inference"]), 3),
        }

        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values(["model_family", "mAP50-95"], ascending=[True, False])

    df.to_excel(args.output, index=False)
    print(f"\nSaved benchmark table to {args.output}")
