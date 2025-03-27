import numpy as np
import os
from dtaidistance import dtw_ndim


def interpolate_trajectories(trajectories):
    result = []
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
        result.append(new_trajectory.reshape(int(len(new_trajectory) / 3), 3))

    return result


cluster1 = [
    np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_trajectories",
            "Video-3.mp4.npy",
        )
    ),
    np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_trajectories",
            "Video-3.mp4.npy",
        )
    ),
    np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_trajectories",
            "Video-3.mp4.npy",
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

cluster2 = [
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


def avg_intracluster_distance(cluster):
    result = 0
    n = 0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            if i == j:
                continue
            trajectory1 = cluster[i]
            trajectory2 = cluster[j]
            distance = dtw_ndim.distance(trajectory1[:, :1], trajectory2[:, :1])
            result += distance
            n += 1
    return result / n


def avg_intercluster_distance(cluster1, cluster2):
    result = 0
    n = 0
    for trajectory1 in cluster1:
        for trajectory2 in cluster2:
            distance = dtw_ndim.distance(trajectory1[:, :1], trajectory2[:, :1])
            result += distance
            n += 1
    return result / n


if __name__ == "__main__":
    all_trajectories = interpolate_trajectories(cluster1 + cluster2)

    cluster1 = all_trajectories[: len(cluster1)]
    cluster2 = all_trajectories[len(cluster1) :]

    intracluster_dist1 = avg_intracluster_distance(cluster1)
    intracluster_dist2 = avg_intracluster_distance(cluster2)
    intercluster_dist = avg_intercluster_distance(cluster1, cluster2)
    db_index = (intracluster_dist1 + intracluster_dist2) / intercluster_dist
    print(
        db_index,
    )
