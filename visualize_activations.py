import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def preprocess_image(image_path):
    image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (400, 200))[100:, :]
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    blurred = cv2.GaussianBlur(image, (15, 15), 10)
    median_intensity = np.median(blurred)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
    canny_edges = cv2.Canny(blurred, 
                             threshold1=lower_threshold, 
                             threshold2=upper_threshold, 
                             apertureSize=5)
    return canny_edges

def create_activation_animation(model_path, csv_path, images_dir):
    # Load model and data
    model = load_model(model_path)
    csv_data = pd.read_csv(csv_path).to_numpy()

    # Prepare animation data
    animation_data = []

    for index in tqdm(range(0, 254)):
        image_memory_stack = []
        for i in range(0, 4):
            current_index = index - i
            image_filename = f"{current_index}.png"
            image_path = os.path.join(images_dir, image_filename)
            if current_index >= 0:
                image_memory_stack.append(preprocess_image(image_path))
            else:
                image_memory_stack.append(np.zeros((100, 400)))
        
        original_image = cv2.resize(cv2.imread(os.path.join(images_dir, f"{index}.png"), cv2.IMREAD_COLOR), (400, 200))
        combined_image = np.concatenate(image_memory_stack)

        # Reshape image to match model's input shape
        input_image = combined_image.reshape(1, 400, 400, 1)

        # Create activation model
        layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        activation_model = tf.keras.Model(
            inputs=model.input, 
            outputs=[model.get_layer(name).output for name in layer_names]
        )

        # Get activations
        activations = activation_model.predict(input_image, verbose=0)
        results = model.predict(input_image, verbose=0)

        animation_data.append({
            'original_image': original_image, 
            'memory_stack': combined_image, 
            'activations': activations, 
            'timestamp': csv_data[index][2], 
            'x': csv_data[index][0], 
            'y': csv_data[index][1],
            'results': results
        })

    # Create the animation
    fig = plt.figure(figsize=(20, 10))

    # Subplots layout
    gs = fig.add_gridspec(3, 3)
    
    # Original image subplot
    ax_original = fig.add_subplot(gs[0, 1])
    ax_original.set_title('Original Image')
    im_original = ax_original.imshow(animation_data[0]['original_image'])
    ax_original.axis('off')

    # Memory stack subplot
    ax_memory = fig.add_subplot(gs[1, 0])
    ax_memory.set_title('Memory Stack')
    im_memory = ax_memory.imshow(animation_data[0]['memory_stack'], cmap='gray')
    ax_memory.axis('off')

    # Coordinate plane subplot
    ax_coords = fig.add_subplot(gs[0, 0])
    ax_coords.set_title('Original controls')
    ax_coords.set_xlim(-1, 1)
    ax_coords.set_ylim(-1, 1)
    ax_coords.set_aspect('equal')
    ax_coords.set_axis_off()
    ax_coords.invert_yaxis()
    ax_coords.set_xlabel('X')
    ax_coords.set_ylabel('Y')
    ax_coords.grid(True)
    circle = plt.Circle((0, 0), 0.99, fill=False, color='gray')
    ax_coords.add_artist(circle)
    point, = ax_coords.plot(animation_data[0]['x'], animation_data[0]['y'], 'ro')

    # Coordinate plane subplot
    ax_coords_predict = fig.add_subplot(gs[0, 2])
    ax_coords_predict.set_title('Predicted controls')
    ax_coords_predict.set_xlim(-1, 1)
    ax_coords_predict.set_ylim(-1, 1)
    ax_coords_predict.set_aspect('equal')
    ax_coords_predict.set_axis_off()
    ax_coords_predict.invert_yaxis()
    ax_coords_predict.set_xlabel('X')
    ax_coords_predict.set_ylabel('Y')
    ax_coords_predict.grid(True)
    circle = plt.Circle((0, 0), 0.99, fill=False, color='gray')
    ax_coords_predict.add_artist(circle)
    predicted_point, = ax_coords_predict.plot(animation_data[0]['results'][0][0], animation_data[0]['results'][0][1], 'ro')

    # Activation layers subplots
    ax_activations = []
    im_activations = []
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    
    for i in range(5):
        ax = fig.add_subplot(gs[1 if i < 2 else 2, (i + 1) % 3])
        ax.set_title(layer_names[i])
        avg_activation = np.mean(animation_data[0]['activations'][i][0], axis=-1)
        im = ax.imshow(avg_activation, cmap='viridis')
        ax.axis('off')
        ax_activations.append(ax)
        im_activations.append(im)
    
    def init():
        # Update original image
        im_original.set_array(animation_data[0]['original_image'])
        
        # Update memory stack
        im_memory.set_array(animation_data[0]['memory_stack'])
        
        # Update coordinate point
        point.set_data([animation_data[0]['x']], [animation_data[0]['y']])

        predicted_point.set_data([animation_data[0]['results'][0][0]], [animation_data[0]['results'][0][1]])
        
        # Update activation layers
        for i in range(5):
            avg_activation = np.mean(animation_data[0]['activations'][i][0], axis=-1)
            im_activations[i].set_array(avg_activation)
        
        return [im_original, im_memory, point, predicted_point] + im_activations

    def update(frame):
        # Update original image
        im_original.set_array(animation_data[frame]['original_image'])
        
        # Update memory stack
        im_memory.set_array(animation_data[frame]['memory_stack'])
        
        # Update coordinate point
        point.set_data([animation_data[frame]['x']], [animation_data[frame]['y']])

        predicted_point.set_data([animation_data[frame]['results'][0][0]], [animation_data[frame]['results'][0][1]])
        
        # Update activation layers
        for i in range(5):
            avg_activation = np.mean(animation_data[frame]['activations'][i][0], axis=-1)
            im_activations[i].set_array(avg_activation)
        
        return [im_original, im_memory, point, predicted_point] + im_activations

    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        init_func=init,
        frames=len(animation_data), 
        interval=0.02,
        blit=True
    )

    plt.tight_layout()
    plt.show()
    return anim

# Example usage
model_path = 'model.h5'
csv_path = "D:/bachelor arbeit/reduced_data/data.csv"
images_dir = "D:/bachelor arbeit/reduced_data/images"

# Create the animation
anim = create_activation_animation(model_path, csv_path, images_dir)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
anim.save('scatter.gif', writer=writer)