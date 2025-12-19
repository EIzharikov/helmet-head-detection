def analyze_results(model, config_path, dataset_path):
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

    # Validate the model on validation set
    print("\nRunning validation on validation set...")
    val_results = model.val()

    print("\n" + "=" * 50)
    print("VALIDATION METRICS")
    print("=" * 50)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")

    # Test on test set by creating a temporary YAML pointing to test set
    print("\n" + "=" * 50)
    print("Testing on test set...")
    print("=" * 50)
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)
    test_img_dir = dataset_path / "helm" / "helm" / "images" / "test"
    test_label_dir = dataset_path / "helm" / "helm" / "labels" / "test"
    if test_img_dir.exists() and test_label_dir.exists():
        test_config = data_config.copy()
        test_config["val"] = str(test_img_dir)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
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

    print("\n" + "=" * 50)
    print("Training and evaluation completed!")
    print("=" * 50)
