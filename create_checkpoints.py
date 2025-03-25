import numpy as np
import pandas as pd
import cv2

if __name__ == "__main__":
    data = pd.read_csv("reduced_data/data.csv").to_numpy()[:15008]

    checkpoints = np.array([])

    index = 0
    reached_next_run = True
    while True:
        if index >= len(data):
            np.save("checkpoints.npy", checkpoints)
            exit()
        if data[index, 2] == 0:
            reached_next_run = True
        if not reached_next_run:
            index += 1
            continue

        # Read a new frame
        frame = cv2.imread("reduced_data/images/" + str(index) + ".png")

        cv2.imshow("Checkpoints", frame)

        while True:
            key = cv2.waitKey(33)

            if key == -1:
                continue
            elif key == 2:
                index -= 2
                break
            elif key == 3:
                break
            elif key == 27:
                print(checkpoints)
                np.save("checkpoints.npy", checkpoints)
                exit()
            elif key == 32:
                checkpoints = np.append(checkpoints, [index])
                reached_next_run = False
                break
        index += 1
