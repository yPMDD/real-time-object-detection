import os
import shutil
import random
from pathlib import Path

def process_and_split():
    # 1. Setup Paths
    raw_images_dir = Path("data/raw/emsi_logo/train/images")
    raw_labels_dir = Path("data/raw/emsi_logo/train/labels")
    dest_base = Path("data/processed")
    
    # 2. Ratios
    train_ratio = 0.8
    val_ratio = 0.1
    # test_ratio will be the remaining 0.1
    
    # Clean old data in processed to avoid mixing
    if dest_base.exists():
        shutil.rmtree(dest_base)
    
    # 3. Get and shuffle images
    images = [f for f in os.listdir(raw_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"Total images: {total}")
    for name, img_list in splits.items():
        print(f"  - {name.capitalize()}: {len(img_list)} images")

    def move_and_remap(image_list, split_name):
        img_dest = dest_base / "images" / split_name
        lbl_dest = dest_base / "labels" / split_name
        
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for img_name in image_list:
            # Move Image
            shutil.copy(raw_images_dir / img_name, img_dest / img_name)
            
            # Process label
            lbl_name = img_name.rsplit('.', 1)[0] + ".txt"
            raw_lbl_path = raw_labels_dir / lbl_name
            
            if raw_lbl_path.exists():
                with open(raw_lbl_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if parts[0] == '0':
                        parts[0] = '4'  # Remap to EMSI_logo ID
                    new_lines.append(" ".join(parts) + "\n")
                
                with open(lbl_dest / lbl_name, 'w') as f:
                    f.writelines(new_lines)

    # 4. Execute
    for name, img_list in splits.items():
        move_and_remap(img_list, name)
    
    print("\nDataset split and remapped successfully!")

if __name__ == "__main__":
    process_and_split()
