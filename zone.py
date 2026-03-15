import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point: [{x}, {y}]")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Define Zones", img)

cap = cv2.VideoCapture("assets/sample-video.mp4")
ret, img = cap.read()
cap.release()

cv2.imshow("Define Zones", img)
cv2.setMouseCallback("Define Zones", click_event)

print("Click the corners of your zones. Press 'q' when done.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nAll points:")
print(points)
