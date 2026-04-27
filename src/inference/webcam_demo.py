import cv2
from ultralytics import YOLO

def run_demo():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam demo... Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        
        if not success:
            print("Error: Could not read frame.")
            break
        results = model(frame, stream=True, device=0)
        for r in results:
            annotated_frame = r.plot()

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
