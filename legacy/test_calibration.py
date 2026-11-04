import cv2
import numpy as np

H = np.load("homography.npy")

img = cv2.imread("court.jpg")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = np.array([x, y, 1.0])
        world = H @ pixel
        world = world / world[2]
        print(f"Clicked pixel ({x}, {y}) -> real-world ({world[0]:.2f}m, {world[1]:.2f}m)")

        cv2.circle(img, (x, y), 5, (255, 100, 0), -1)
        cv2.imshow("Test Calibration", img)

print("Click anywhere on the court to test the mapping. Press any key to quit.")
cv2.imshow("Test Calibration", img)
cv2.setMouseCallback("Test Calibration", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()