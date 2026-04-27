import cv2
import os
import time

def collect_images():
    # Path where we will save the raw logo images
    save_path = "data/raw/emsi_logo"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    # Open the webcam (using index 0 for Iriun)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- EMSI Logo Data Collector (Real-World) ---")
    print("Instructions:")
    print("1. Point your camera at the EMSI logo (on paper, screen, or building).")
    print("2. Press 's' to SAVE an image.")
    print("3. Press 'q' to QUIT.")
    print("---------------------------------------------")

    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow("Capture Real-World Examples - Press 's'", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = int(time.time())
            filename = f"real_emsi_{timestamp}_{count}.jpg"
            file_path = os.path.join(save_path, filename)
            cv2.imwrite(file_path, frame)
            print(f"Saved Real Image: {file_path}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! You collected {count} real-world images.")

if __name__ == "__main__":
    collect_images()
