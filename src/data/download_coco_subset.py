import fiftyone as fo
import fiftyone.zoo as foz
import os

def download_coco_subset():
    # 1. Define classes and limit
    classes = ["person", "bottle", "cell phone", "mouse"]
    max_samples = 100 # 100 images per class
    
    save_path = "data/raw/coco_subset"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Starting download for classes: {classes}")
    
    # 2. Download from the zoo
    # We download a subset that contains ANY of our target classes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation", # Using validation split for faster download (smaller but still big)
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples * len(classes),
    )

    # 3. Remap Class IDs to match our project
    # Project IDs: 0:person, 1:bottle, 2:phone, 3:mouse
    class_map = {
        "person": "person",
        "bottle": "bottle",
        "cell phone": "phone",
        "mouse": "mouse"
    }
    
    print(f"Dataset fields: {dataset.get_field_schema().keys()}")
    label_field = "ground_truth" if "ground_truth" in dataset.get_field_schema() else "detections"
    print(f"Using label field: {label_field}")
    
    # Filter the labels to only include our 4 classes
    view = dataset.filter_labels(label_field, fo.ViewField("label").is_in(classes))

    # 4. Export in YOLOv8 Format
    print(f"Exporting to {save_path}...")
    view.export(
        export_dir=save_path,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=["person", "bottle", "phone", "mouse", "EMSI_logo"] # Define all 5 project classes
    )

    print("\nDownload and Export complete!")

if __name__ == "__main__":
    download_coco_subset()
