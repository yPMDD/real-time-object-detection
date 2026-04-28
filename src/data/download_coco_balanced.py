import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil

def download_balanced_coco():
    # 1. Define our target classes and exactly how many we want
    target_distribution = {
        "cell phone": 350,
        "bottle": 350,
        "mouse": 350,
        "person": 350
    }
    
    save_path = "data/raw/coco_subset"
    
    # Clean up old database if it exists
    if fo.dataset_exists("balanced_coco_dataset"):
        fo.delete_dataset("balanced_coco_dataset")
        
    master_dataset = fo.Dataset("balanced_coco_dataset")
    master_dataset.persistent = False

    # 2. Download individually to force balance
    for cls, count in target_distribution.items():
        print(f"\n--- Downloading exactly {count} images containing '{cls}' ---")
        
        temp_name = f"temp_{cls.replace(' ', '_')}"
        if fo.dataset_exists(temp_name):
            fo.delete_dataset(temp_name)
            
        # We use the 'train' split because it has much more data than 'validation'
        cls_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=[cls],
            max_samples=count,
            dataset_name=temp_name
        )
        
        master_dataset.add_collection(cls_dataset)
        fo.delete_dataset(temp_name)

    print(f"\nTotal images collected: {len(master_dataset)}")

    # 3. Filter out all the random other COCO classes (like 'dog', 'car')
    class_map = {
        "person": "person",
        "bottle": "bottle",
        "cell phone": "phone",
        "mouse": "mouse"
    }
    
    label_field = "ground_truth" if "ground_truth" in master_dataset.get_field_schema() else "detections"
    
    print("Filtering and mapping labels...")
    view = master_dataset.filter_labels(label_field, fo.ViewField("label").is_in(list(class_map.keys())))
    
    # 4. Map 'cell phone' to 'phone'
    view = view.map_labels(label_field, class_map)

    # Clean the old export directory
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # 5. Export for YOLO
    print(f"Exporting perfectly balanced dataset to {save_path}...")
    view.export(
        export_dir=save_path,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=["person", "bottle", "phone", "mouse", "EMSI_logo"]
    )

    print("\nDownload and Export complete!")

if __name__ == "__main__":
    download_balanced_coco()
