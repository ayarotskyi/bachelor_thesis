import matplotlib
matplotlib.use('Agg')
import json
import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


from keras import Sequential
from keras.api.layers import Flatten, Dense, Lambda, Cropping2D
from keras.api.layers import Conv2D
import os

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

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

def prepare_image_data(
    image_dir, 
    csv_path, 
    augment=True,
    test_size=0.2,
    random_state=42
):
    # 1. Read CSV file
    df = pd.read_csv(csv_path)
    
    # 2. Prepare image and label lists
    images = []
    labels = []
    
    # 3. Load and preprocess images
    for index, row in df.iterrows():
        # Construct image filename (assuming image name matches row index)
        image_filename = f"{index}.png"  # Adjust extension as needed
        image_path = os.path.join(image_dir, image_filename)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_filename} not found. Skipping.")
            continue
        
        # Read and preprocess image
        try:
            image_array = preprocess_image(image_path)
            
            images.append(image_array)
            labels.append([float(row['x']), float(row['y'])])  # Adjust column name as needed
        
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
        
    # Combine original and flipped images
    x_augmented = []
    y_augmented = []
    
    for img, label in zip(images, labels):
        # Original image
        x_augmented.append(img)
        y_augmented.append(label)
        
        # Horizontally flipped image
        flipped_img = np.fliplr(img)
        x_augmented.append(flipped_img)
        y_augmented.append([label[1], label[0]])
    
    x = np.array(x_augmented)
    y = np.array(y_augmented)
    
    return x, y

if __name__ == "__main__":
    # Example of how to use the function
	x, y = prepare_image_data(
		image_dir="D:/bachelor arbeit/reduced_data/images",
		csv_path="D:/bachelor arbeit/reduced_data/data.csv",
		augment=True
	)
	model = Sequential()
	model.add(Lambda(lambda x: (x/255), input_shape = (100, 400)))
	model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv1'))
	model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv2'))
	model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv3'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv4'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv5')) 
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(2))

	model.compile(loss = 'mse', optimizer = 'adam')
	print("starting training")
	history_object = model.fit(x=x, y=y, batch_size=32,validation_split=0.2, shuffle=True)
	model.save('model_nvid_angle.h5')
	outfile = open('./model_nvid_angle.json', 'w') 
	json.dump(model.to_json(), outfile)
	outfile.close()
	model.save_weights('model_nvid_angle_weights.h5')

	# print(history_object.history.keys())
	# fig = plt.figure()
	# plt.plot(history_object.history['loss'])
	# plt.plot(history_object.history['val_loss'])
	# plt.ylabel('Mean Squared Error Loss')
	# plt.xlabel('Epoch')
	# fig.savefig('test_val_ac_angle1.png')