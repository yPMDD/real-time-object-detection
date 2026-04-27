from ultralytics import YOLO
import os

def train_model():
    # 1. Load the base model
    # We use 'yolov8n.pt' as a starting point (Transfer Learning)
    model = YOLO("yolov8n.pt")

    # 2. Path to our dataset configuration
    # Note: Use an absolute path or relative to the project root
    dataset_config = "data/dataset.yaml"
    
    # Ensure the path is correct
    if not os.path.exists(dataset_config):
        print(f"Error: Could not find {dataset_config}")
        return

    print("--- Starting Training ---")
    print(f"Dataset: {dataset_config}")
    print("Device: GPU (0)")

    # 3. Train the model
    # epochs: Number of times to see the whole dataset
    # imgsz: Image size (standard is 640)
    # batch: Number of images per step (adjust if you get 'out of memory')
    results = model.train(
        data=dataset_config,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name="emsi_object_detector", # Name of the folder where results will be saved
        exist_ok=True
    )

    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/emsi_object_detector/weights/best.pt")

if __name__ == "__main__":
    train_model()
