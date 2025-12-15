from ultralytics import YOLO
from pathlib import Path
import kagglehub
import yaml
import tempfile
import os

def main():
    path = Path(kagglehub.dataset_download("vodan37/yolo-helmethead"))
    yaml_path = path / "helm" / "helm" / "helm.yaml"
    model = YOLO('yolov5n.pt')
    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        cache=False
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    
    # Validate the model on validation set
    print("\nRunning validation on validation set...")
    val_results = model.val()
    
    print("\n" + "="*50)
    print("VALIDATION METRICS")
    print("="*50)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    
    # Test on test set by creating a temporary YAML pointing to test set
    print("\n" + "="*50)
    print("Testing on test set...")
    print("="*50)
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    test_img_dir = path / "helm" / "helm" / "images" / "test"
    test_label_dir = path / "helm" / "helm" / "labels" / "test"
    if test_img_dir.exists() and test_label_dir.exists():
        test_config = data_config.copy()
        test_config['val'] = str(test_img_dir)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(test_config, tmp_file)
            tmp_yaml_path = tmp_file.name
        
        try:
            test_results = model.val(data=tmp_yaml_path)
            
            print(f"Test mAP50: {test_results.box.map50:.4f}")
            print(f"Test mAP50-95: {test_results.box.map:.4f}")
            print(f"Test Precision: {test_results.box.mp:.4f}")
            print(f"Test Recall: {test_results.box.mr:.4f}")
        finally:
            os.unlink(tmp_yaml_path)
    else:
        print("Test set directory not found. Skipping test set evaluation.")
    
    print("\n" + "="*50)
    print("Training and evaluation completed!")
    print("="*50)

if __name__ == '__main__':
    main()