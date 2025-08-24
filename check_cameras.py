import cv2

def find_cameras():
    """Tests camera indices 0 through 4 to see which are available."""
    print("Searching for available cameras...")
    for i in range(5):
        print(f"--- Checking index {i} ---")
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            print(f"✅ SUCCESS: Camera found at index {i}")
            cap.release()
        else:
            print(f"❌ FAILURE: No camera at index {i}")
    print("Search complete.")

if __name__ == "__main__":
    find_cameras()
