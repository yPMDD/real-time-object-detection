import os
import shutil
import random
from pathlib import Path

def merge_coco_into_processed():
    # 1. Setup Paths
    coco_images_dir = Path("data/raw/coco_subset/images/val")
    coco_labels_dir = Path("data/raw/coco_subset/labels/val")
    
    dest_base = Path("data/processed")
    
    # 2. Ratios for the COCO subset
    train_ratio = 0.8
    # Remaining 0.2 goes to val
    
    # 3. Get all COCO images
    coco_images = [f for f in os.listdir(coco_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(coco_images)
    
    split_idx = int(len(coco_images) * train_ratio)
    train_imgs = coco_images[:split_idx]
    val_imgs = coco_images[split_idx:]

    print(f"Merging COCO subset...")
    print(f"Found {len(coco_images)} COCO images.")
    print(f"Adding {len(train_imgs)} to Train and {len(val_imgs)} to Val.")

    def copy_files(image_list, split_name):
        img_dest = dest_base / "images" / split_name
        lbl_dest = dest_base / "labels" / split_name
        
        # Ensure destinations exist (they should already exist from split_dataset.py)
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for img_name in image_list:
            # Copy Image
            shutil.copy(coco_images_dir / img_name, img_dest / img_name)
            
            # Copy Label
            lbl_name = img_name.rsplit('.', 1)[0] + ".txt"
            if (coco_labels_dir / lbl_name).exists():
                shutil.copy(coco_labels_dir / lbl_name, lbl_dest / lbl_name)
            else:
                print(f"Warning: Label missing for COCO image {img_name}")

    # 4. Execute Merge
    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    
    print("\nMerge complete! Your dataset is now balanced.")
    
    # Summary of final counts
    for split in ["train", "val", "test"]:
        count = len(os.listdir(dest_base / "images" / split)) if (dest_base / "images" / split).exists() else 0
        print(f"Final {split.capitalize()} set size: {count} images")

if __name__ == "__main__":
    merge_coco_into_processed()
