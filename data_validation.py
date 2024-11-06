import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
import sys
import os
from PIL import Image
import glob

import pandas as pd

def load_data_and_images(directory):
    """
    Load numpy array and image files from the specified directory.
    
    Parameters:
    directory: str, path to the directory containing data.npy and images subdirectory
    
    Returns:
    tuple: (numpy array, list of image paths)
    """
    # Load numpy array
    data_path = os.path.join(directory, 'data.npy')
    try:
        data = np.load(data_path)
        
        # Verify array structure
        required_fields = {'x': np.float64, 'y': np.float64, 'timestamp': 'datetime64[ms]'}
        for field, dtype in required_fields.items():
            if field not in data.dtype.names:
                raise ValueError(f"Missing required field: {field}")
            if data.dtype[field] != np.dtype(dtype):
                raise ValueError(f"Incorrect dtype for {field}. Expected {dtype}, got {data.dtype[field]}")
        
        # Sort data by timestamp
        data = np.sort(data, order='timestamp')
        
        # Load image paths
        images_dir = os.path.join(directory, 'images')
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get sorted list of jpg files
        image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        
        if len(image_paths) != len(data):
            raise ValueError(f"Number of images ({len(image_paths)}) doesn't match data length ({len(data)})")
        
        return data, image_paths
    
    except Exception as e:
        print(f"Error loading data from {directory}: {str(e)}")
        sys.exit(1)

def create_synchronized_animation(data, image_paths):
    """
    Creates a synchronized animation of points moving on a circle and corresponding images.
    
    Parameters:
    data: numpy structured array with fields 'x', 'y', and 'timestamp'
    image_paths: list of paths to image files
    """
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(15, 7))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Circular animation subplot
    ax_circle = fig.add_subplot(gs[0])
    circle = plt.Circle((0, 0), 1, fill=False, color='gray')
    ax_circle.add_artist(circle)
    ax_circle.set_xlim(-1.2, 1.2)
    ax_circle.set_ylim(-1.2, 1.2)
    ax_circle.set_aspect('equal')
    ax_circle.grid(True)
    ax_circle.set_title('Point Animation')
    
    # Initialize point and trail
    point, = ax_circle.plot([], [], 'ro', markersize=8)
    trail, = ax_circle.plot([], [], 'r-', alpha=0.3)
    trail_x, trail_y = [], []
    
    # Image subplot
    ax_image = fig.add_subplot(gs[1])
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title('Camera Feed')
    
    # Load first image to get dimensions
    first_image = plt.imread(image_paths[0])
    img_display = ax_image.imshow(first_image)
    
    # Add timestamp display
    timestamp_text = ax_circle.text(0.02, 1.1, '', transform=ax_circle.transAxes)
    
    def init():
        """Initialize animation"""
        point.set_data([], [])
        trail.set_data([], [])
        img_display.set_array(np.zeros_like(first_image))
        timestamp_text.set_text('')
        return point, trail, img_display, timestamp_text
    
    def animate(frame):
        """Animation function called for each frame"""
        # Update trail
        trail_x.append(data[frame]['x'])
        trail_y.append(data[frame]['y'])
        trail.set_data(trail_x, trail_y)
        
        # Update current point
        point.set_data([data[frame]['x']], [data[frame]['y']])
        
        # Update image
        new_image = plt.imread(image_paths[frame])
        img_display.set_array(new_image)
        
        # Update timestamp
        timestamp = pd.Timestamp(data[frame]['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        timestamp_text.set_text(f'Timestamp: {timestamp}')
        
        return point, trail, img_display, timestamp_text
    
    # Calculate frame intervals
    intervals = np.diff(data['timestamp'].astype(np.int64)) / 1e6  # Convert to milliseconds
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        animate,
        init_func=init,
        frames=len(data),
        interval=intervals[0],  # Start with first interval
        blit=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim

def main():
    parser = argparse.ArgumentParser(
        description='Create synchronized animation of points and images based on timestamp data.'
    )
    parser.add_argument('directory', 
                      help='Directory containing data.npy and images subdirectory')
    
    args = parser.parse_args()
    
    # Load data and images
    data, image_paths = load_data_and_images(args.directory)
    
    # Create and show animation
    anim = create_synchronized_animation(data, image_paths)

if __name__ == "__main__":
    main()