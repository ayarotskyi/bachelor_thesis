import json
from keras import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
import keras
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib as plt

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

def prepare_image_data_generator(
    image_dir,
    csv_path,
    batch_size=32
):
    # 1. Read CSV file
    df = pd.read_csv(csv_path)
    df_array = df.to_numpy()

    def data_generator():
        images = []
        labels = []

        for index, row in df.iterrows():
            try:
                # Read and preprocess image memory stack
                image_memory_stack = []
                overflows_starting_timestamp = False
                for i in range(0, 4):
                    current_index = index - i
                    image_filename = f"{current_index}.png"
                    image_path = os.path.join(image_dir, image_filename)

                    if (
                        not overflows_starting_timestamp
                        and current_index > 0
                        and int(df_array[current_index][2]) > 0
                        and os.path.exists(image_path)
                    ):
                        image_memory_stack.append(preprocess_image(image_path))
                    else:
                        overflows_starting_timestamp = True
                        image_memory_stack.append(np.zeros((100, 400)))

                # Combine memory stack and add to list
                combined_image = np.concatenate(image_memory_stack)
                images.append(combined_image)
                labels.append([float(row['x']), float(row['y'])])  # Adjust column names as needed

                # Augmentation: flipped image
                flipped_memory_stack = np.fliplr(combined_image)
                images.append(flipped_memory_stack)
                labels.append([float(row['y']), float(row['x'])])

                # Yield batch if size matches batch_size
                if len(images) >= batch_size:
                    yield np.array(images), np.array(labels)
                    images = []
                    labels = []

            except Exception as e:
                print(f"Error processing: {e}")

        # Yield any remaining data
        if images:
            yield np.array(images), np.array(labels)

    return data_generator()

if __name__ == "__main__":
    # Example of how to use the function
    data_generator = prepare_image_data_generator(
		image_dir="D:/bachelor arbeit/reduced_data/images",
		csv_path="D:/bachelor arbeit/reduced_data/data.csv",
	)
    model = Sequential()
    model.add(Lambda(lambda x: (x/255), input_shape = (400, 400, 1)))
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

    history= model.fit(x=data_generator, 
                       batch_size=32,
                       shuffle=True, 
                       epochs=100, 
                       callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10,  # Stops if no improvement for 10 epochs
            restore_best_weights=True
        )
    ])
    model.save('model.h5')
    outfile = open('./model_nvid_angle.json', 'w') 
    json.dump(model.to_json(), outfile)
    outfile.close()
    model.save_weights('model.weights.h5')  

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()