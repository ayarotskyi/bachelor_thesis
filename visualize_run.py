import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import matplotlib.animation as animation

directory = "results/1742898509661"
if __name__ == "__main__":
    data = np.load(os.path.join(directory, "data.npy"))
    np.savetxt(os.path.join(directory, "data.csv"), data, delimiter=",")
    animation_data = []
    for index, value in enumerate(data):
        animation_data.append(
            {
                "image": cv2.imread(os.path.join(directory, str(index) + ".jpg")),
                "x": value[0],
                "y": value[1],
            }
        )

    fig = plt.figure(figsize=(20, 10))

    gs = fig.add_gridspec(3, 3)

    # Original image subplot
    ax_original = fig.add_subplot(gs[0, 1])
    ax_original.set_title("Original Image")
    im_original = ax_original.imshow(animation_data[0]["image"])
    ax_original.axis("off")

    # Coordinate plane subplot
    ax_coords = fig.add_subplot(gs[0, 0])
    ax_coords.set_title("Original controls")
    ax_coords.set_xlim(-1, 1)
    ax_coords.set_ylim(-1, 1)
    ax_coords.set_aspect("equal")
    ax_coords.set_axis_off()
    ax_coords.invert_yaxis()
    ax_coords.set_xlabel("X")
    ax_coords.set_ylabel("Y")
    ax_coords.grid(True)
    circle = plt.Circle((0, 0), 0.99, fill=False, color="gray")
    ax_coords.add_artist(circle)
    (point,) = ax_coords.plot(animation_data[0]["x"], animation_data[0]["y"], "ro")

    def init():
        # Update original image
        im_original.set_array(animation_data[0]["image"])

        # Update coordinate point
        point.set_data([animation_data[0]["x"]], [animation_data[0]["y"]])

        return [im_original, point]

    def update(frame):
        # Update original image
        im_original.set_array(animation_data[frame]["image"])

        # Update coordinate point
        point.set_data([animation_data[frame]["x"]], [animation_data[frame]["y"]])

        return [im_original, point]

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(animation_data),
        interval=1000,
        blit=True,
    )

    plt.tight_layout()
    plt.show()
