import cv2
import numpy as np

# Real-world court points, in meters
# Origin (0,0) is the near-left baseline corner
REAL_POINTS = np.array([
    [0, 0],        # baseline corner, near left
    [8.23, 0],      # baseline corner, near right
    [0, 5.485],     # service line corner, left
    [8.23, 5.485]   # service line corner, right
], dtype=np.float32)

clicked_pixels = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
        clicked_pixels.append([x, y])
        print(f"Point {len(clicked_pixels)} clicked at pixel ({x}, {y})")

        cv2.circle(img, (x, y), 6, (0, 100, 255), -1)
        cv2.imshow("Court Calibration", img)

        if len(clicked_pixels) == 4:
            compute_homography()

def compute_homography():
    pixel_pts = np.array(clicked_pixels, dtype=np.float32)
    H, status = cv2.findHomography(pixel_pts, REAL_POINTS)
    print("\nHomography matrix:")
    print(H)
    np.save("homography.npy", H)
    print("\nSaved to homography.npy")

# Load your court image
img = cv2.imread("court.jpg")

if img is None:
    print("Could not find court.jpg — make sure it's in the tennis-cv folder")
else:
    print("Click these 4 points in order:")
    print("1. Baseline corner, near left")
    print("2. Baseline corner, near right")
    print("3. Service line corner, left")
    print("4. Service line corner, right")

    cv2.imshow("Court Calibration", img)
    cv2.setMouseCallback("Court Calibration", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()