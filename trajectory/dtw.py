import numpy as np
from dtaidistance import dtw_ndim
from dtaidistance import dtw_ndim_visualisation
from matplotlib import pyplot as plt
import os

trajectories = [
    np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_trajectories",
            "Video-1.mp4.npy",
        )
    ),
    np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_trajectories",
            "Video-2.mp4.npy",
        )
    ),
]

timestamps = set()

for trajectory in trajectories:
    for timestamp in trajectory[:, 2]:
        timestamps.add(timestamp)

all_timestamps = np.sort(np.array(list(timestamps)))


for i in range(len(trajectories)):
    trajectory = trajectories[i]
    new_trajectory = np.array([])
    current_index = 0
    for timestamp in all_timestamps:
        if (
            current_index < len(trajectory)
            and trajectory[current_index][2] == timestamp
        ):
            new_trajectory = np.append(new_trajectory, trajectory[current_index])
            current_index += 1
        else:
            interpolation_start = trajectory[
                current_index - 1 if current_index > 0 else 0
            ]
            interpolation_end = (
                trajectory[current_index]
                if current_index < len(trajectory)
                else trajectory[-1]
            )
            interpolation_multiplier = (
                (timestamp - interpolation_start[2])
                / (interpolation_end[2] - interpolation_start[2])
                if interpolation_end[2] != interpolation_start[2]
                else 0
            )
            new_trajectory = np.append(
                new_trajectory,
                [
                    (interpolation_end[0] - interpolation_start[0])
                    * interpolation_multiplier
                    + interpolation_start[0],
                    (interpolation_end[1] - interpolation_start[1])
                    * interpolation_multiplier
                    + interpolation_start[1],
                    timestamp,
                ],
            )
    trajectories[i] = new_trajectory.reshape(int(len(new_trajectory) / 3), 3)

s1 = trajectories[0][:, :1]
s2 = trajectories[1][:, :1]

d, paths = dtw_ndim.warping_paths(s1, s2, window=25, psi=2)
figure = dtw_ndim_visualisation.plot_warpingpaths(s1, s2, paths)

plt.show()
