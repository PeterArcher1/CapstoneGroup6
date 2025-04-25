import cv2
import numpy as np

# Use the correct function for OpenCV 4.7+
apriltag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()

# Camera Calibration Values
KNOWN_TAG_SIZE = 0.2  # AprilTag size in meters
FOCAL_LENGTH = 1470  # Adjust based on calibration

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    corners, ids, _ = cv2.aruco.detectMarkers(gray, apriltag_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Compute center of the tag
            corner_pts = corners[i][0]
            center_x = int((corner_pts[0][0] + corner_pts[2][0]) / 2)
            center_y = int((corner_pts[0][1] + corner_pts[2][1]) / 2)
            center = (center_x, center_y)

            # Compute error from frame center in pixels
            error_x = center_x - frame_center_x
            error_y = center_y - frame_center_y

            # Compute perceived width of the tag
            perceived_width = np.linalg.norm(corner_pts[0] - corner_pts[1])

            # Compute distance using the pinhole model
            if perceived_width > 0:
                distance = (KNOWN_TAG_SIZE * FOCAL_LENGTH) / perceived_width
                distance /= 3.048  # Convert meters to feet

                # Compute pixel-to-meter ratio
                px_to_m = KNOWN_TAG_SIZE / perceived_width

                # Convert pixel errors to meters
                error_x_m = error_x * px_to_m
                error_y_m = error_y * px_to_m
            else:
                distance = 0
                error_x_m, error_y_m = 0, 0

            # Draw center point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Display Tag ID, error, and distance
            cv2.putText(frame, f"ID: {ids[i][0]}", (center_x - 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Error: ({error_x}px, {error_y}px)", (center_x + 10, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Error: ({error_x_m:.3f}m, {error_y_m:.3f}m)", (center_x + 10, center_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}m", (center_x + 10, center_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw frame center
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

    # Show the frame
    cv2.imshow("AprilTag Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
