import cv2
import numpy as np

# Initialize video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Refine the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Extract the dynamic objects (foreground)
    dynamic_objects = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Extract the static background by removing dynamic objects
    static_background = frame.copy()
    static_background[fg_mask > 0] = (0, 0, 0)

    # Show the results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Dynamic Objects", dynamic_objects)
    cv2.imshow("Static Background", static_background)

    # Break loop on key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
