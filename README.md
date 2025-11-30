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


# Extra links:

1. [Kaggle dataset](https://www.kaggle.com/datasets/vodan37/yolo-helmethead/code)