import cv2
from ultralytics import YOLO
import os

def run_custom_demo():
    # 1. Path to your CUSTOM trained model
    # 'best.pt' is the version that performed best during validation
    model_path = "runs/detect/emsi_object_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find custom model at {model_path}")
        print("Make sure you have completed the training step.")
        return

    # 2. Load the custom model
    print(f"Loading custom model: {model_path}")
    model = YOLO(model_path)
    
    # 3. Open the webcam (Iriun)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- EMSI CUSTOM OBJECT DETECTOR IS LIVE ---")
    print("Detecting: Person, Bottle, Phone, Mouse, EMSI_Logo")
    print("Press 'q' to quit.")
    print("--------------------------------------------")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run inference using your GPU (device=0)
        # We set conf=0.5 to only show detections we are at least 50% sure about
        results = model(frame, stream=True, device=0, conf=0.5)

        for r in results:
            annotated_frame = r.plot()

        # Display the live feed with your custom detections
        cv2.imshow("EMSI Real-Time AI System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_custom_demo()
