# Helmet detection

<img src=images/helmet_detection_example_1.jpg width=44% />
<img src=images/helmet_detection_example_2.jpg width=50% />


## Problem Statement

The goal of this project is to develop a computer vision model for **helmet detection on human heads** in images. The model should be able to:

1. Identify whether a helmet is present or absent on a person's head.
2. Localize both the helmet and the head using **bounding boxes**.

This task falls under **object detection**. The model will use images annotated in the YOLO format, where each object has a class label and normalized bounding box coordinates.

The application of such a model is critical for workplace safety, construction, manufacturing, and other environments where wearing a helmet is mandatory.

## Dataset

The dataset is provided in **YOLO format**:

- Images are split into three subsets: **train (70%)**, **validation (20%)**, and **test (10%)**.  
- Each image has a corresponding `.txt` file in the `labels` directory, containing object annotations:  
  - `1` — helmet  
  - `0` — head  
- Bounding boxes are normalized (values from 0.0 to 1.0):  
  - `(x, y)` is the **center** of the box  
  - `width` and `height` are relative to the image size  

Example annotation file for an image:

| ID    | Cordinate 1 | Cordinate 2 | Cordinate 3 | Cordinate 4 |
| ------| -------     | --------    | -------     | --------    |
|   1   | 0.716797    | 0.395833    | 0.216406    | 0.147222    |
|   1   | 0.687109    | 0.379167    | 0.255469    | 0.158333    |
|   1   | 0.420312    | 0.395833    | 0.140625    | 0.166667    |

### Analytics:

To get from dataset, follow next steps:

1. Create venv (recommended python version is 3.10):
    ```sh
    python -m venv venv
    ```
2. Activate environment:
    ```sh
    # Windows
    ./venv/Scripts/activate

    # Linux\Mac
    source venv/bin/activate
    ```
3. Install requirements:
    ```sh
    pip install -r ./requirements.txt
    ```
4. Launch script:
    ```sh
    python ./analyze_dataset.py
    ```

Table 1. Collected analytics data about dataset.

| Split  | Images | Label Files | Min Image Size | Max Image Size | Heads |Helmets |
|--------|----------|---------------|----------------|----------------|---------|-----------|
| Train  | 15887    | 15887         | 107x112        | 7360x4912      | 89181   | 43127     |
| Valid  | 4641     | 4641          | 107x100        | 6000x4000      | 25868   | 12404     |
| Test   | 2261     | 2261          | 107x144        | 6598x3840      | 13217   | 6751      |

## Training

Training is performed using **Ultralytics YOLO models**.  
The dataset is automatically downloaded from Kaggle using `kagglehub` during training.

### Prerequisites

1. Python **3.10** (recommended)
2. CUDA-enabled GPU (optional but recommended)
3. Installed dependencies:

```sh
pip install -r requirements.txt
```
### Launch
Training is launched via a Python CLI script.

```sh
python train.py \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 16
```

| Argument   | Description                           | Example                    |
| ---------- | ------------------------------------- | -------------------------- |
| `--model`  | YOLO model checkpoint or architecture | `yolov8n.pt`, `yolov8s.pt` |
| `--epochs` | Number of training epochs             | `50`                       |
| `--imgsz`  | Input image size                      | `640`                      |
| `--batch`  | Batch size                            | `16`                       |

### Output
Training outputs

After training, all experiment artifacts are saved to:
```sh
runs/detect/
```

## Inference
Inference is performed using a trained YOLO checkpoint (.pt file).

If you want to launch inference, you should use next command:
```sh
python infer.py \
    --weights runs/detect/yolov8n_baseline/weights/best.pt \
    --source path/to/image_or_directory

| Argument    | Description                              | Example                |
| ----------- | ---------------------------------------- | ---------------------- |
| `--weights` | Path to trained YOLO weights             | `best.pt`              |
| `--source`  | Image, directory, or video for inference | `image.jpg`, `images/` |

```

### Inference outputs

Prediction results are saved automatically to:
```sh
runs/detect/predict/
```

### Notes
Inference does not require access to the Kaggle dataset

Only the trained .pt file is required

GPU is optional for inference, but recommended for speed

# Extra links:

1. [Kaggle dataset](https://www.kaggle.com/datasets/vodan37/yolo-helmethead/code)
