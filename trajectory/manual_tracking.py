import cv2
import sys
import numpy as np
import os

video_name = "Video-3.mp4"

if __name__ == "__main__":
    # Read video
    video = cv2.VideoCapture(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), video_name)
    )
    fps = video.get(cv2.CAP_PROP_FPS)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read the first frame
    ret, prev_frame = video.read()
    if not ret:
        print("Error reading video")
        video.release()
        cv2.destroyAllWindows()
        exit()

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    trajectory = []
    timestamps = []

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, gray)

        # Threshold to detect changes
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels (amount of change)
        motion_level = cv2.countNonZero(thresh)

        # Define motion threshold (adjustable)
        MOTION_THRESHOLD = 500  # Tune this based on your video

        if motion_level > MOTION_THRESHOLD:
            prev_gray = gray  # Update previous frame
        else:
            continue

        allow_continue = False

        def addTrajectoryPoint(event, x, y, flags, param):
            global allow_continue
            # param is the array i from below
            if event == cv2.EVENT_LBUTTONDOWN:
                trajectory.append((x, y))
                timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))
                allow_continue = True

        # Display result
        if len(trajectory) > 0:
            for index in range(len(trajectory) - 1):
                cv2.line(
                    frame, trajectory[index], trajectory[index + 1], (0, 255, 0), 9
                )
        cv2.imshow("Tracking", frame)

        cv2.setMouseCallback("Tracking", addTrajectoryPoint)

        while not allow_continue:
            cv2.waitKey(10)

    array = np.column_stack((trajectory, timestamps))

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    np.save(os.path.join(results_path, video_name + ".npy"), array)
