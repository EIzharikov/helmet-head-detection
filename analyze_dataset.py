from pathlib import Path

import kagglehub
from PIL import Image
from tqdm import tqdm


def analyze_dataset(dataset_path: str):
    splits = ["train", "valid", "test"]
    class_map = {0: "head", 1: "helmet"}

    summary = {}

    for split in splits:
        img_dir = Path(dataset_path) / "helm" / "helm" / "images" / split
        label_dir = Path(dataset_path) / "helm" / "helm" / "labels" / split

        img_files = sorted(list(img_dir.glob("*.jpg")))
        label_files = sorted(list(label_dir.glob("*.txt")))

        print(
            f"\nSplit '{split}': {len(img_files)} images, {len(label_files)} label files"
        )

        widths, heights = [], []
        obj_counts = {cls: 0 for cls in class_map.values()}
        obj_sizes = {cls: {"widths": [], "heights": []} for cls in class_map.values()}

        for img_file, label_file in tqdm(
            zip(img_files, label_files), total=min(len(img_files), len(label_files))
        ):
            with Image.open(img_file) as img:
                w, h = img.size
            widths.append(w)
            heights.append(h)

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x, y, bw, bh = parts
                    cls_id = int(cls_id)
                    obj_counts[class_map[cls_id]] += 1
                    obj_sizes[class_map[cls_id]]["widths"].append(float(bw))
                    obj_sizes[class_map[cls_id]]["heights"].append(float(bh))

        summary[split] = {
            "num_images": len(img_files),
            "image_widths": widths,
            "image_heights": heights,
            "object_counts": obj_counts,
            "object_sizes": obj_sizes,
        }

        print(
            f"Image sizes: min {min(widths)}x{min(heights)}, max {max(widths)}x{max(heights)}"
        )
        print("Object counts:", obj_counts)

    return summary


def main():
    path = kagglehub.dataset_download("vodan37/yolo-helmethead")
    print("Path to dataset files:", path)

    analyze_dataset(path)
    print("\nDataset analysis completed.")


if __name__ == "__main__":
    main()
