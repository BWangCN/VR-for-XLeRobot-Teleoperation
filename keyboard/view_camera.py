"""
Simple camera viewer to identify and test cameras.
Usage: python view_camera.py
Then enter the camera index (0 or 1) when asked.

Press Q in the video window to quit.
"""

import sys

try:
    import cv2
except ImportError:
    print("OpenCV not found. Run: pip install opencv-python")
    sys.exit(1)


def main():
    idx = input("Enter camera index to view (e.g. 0 or 1): ").strip()
    try:
        idx = int(idx)
    except ValueError:
        print("Please enter a number.")
        sys.exit(1)

    # Use DSHOW backend on Windows — MSMF can silently fail to open cameras
    print(f"Opening camera {idx} with DSHOW backend...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Could not open camera {idx}. Try the other index.")
        sys.exit(1)

    print("Camera opened! A window should appear.")
    print("Press Q in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Label the window with the camera index
        cv2.putText(frame, f"Camera {idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(f"Camera {idx} — press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
