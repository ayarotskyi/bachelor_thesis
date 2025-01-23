import json
from keras import Sequential
from keras.layers import Flatten, Dense, Lambda, ConvLSTM2D, BatchNormalization
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

def get_desired_frames_mask(ms_array, desired_frame_rate):
    mask = np.ones(len(ms_array), dtype=bool)
    max_ms_between_elements = (1 / desired_frame_rate) * 1000
    prev_index = 0
    for i in range(1, len(ms_array)):
        if ms_array[i] > ms_array[prev_index] and ms_array[i] - ms_array[prev_index] < max_ms_between_elements:
            mask[i] = False
        else:
            prev_index = i
    return mask

def prepare_image_data_generator(
    image_dir,
    csv_path,
    batch_size=32,
    test_split=0.2
):
    # 1. Read CSV file
    array = pd.read_csv(csv_path).to_numpy()
    timestamp_array = array[:, 2]
    array = np.column_stack((array, np.arange(len(array))))
    array = array[get_desired_frames_mask(array[:, 2], 17)]
    np.random.shuffle(array)
    split_index = int(len(array)*(1 - test_split))
    train_array, test_array = array[:split_index], array[split_index:]

    def data_generator(array, shuffle):
        def returning_generator():
            while True:  # Add an infinite loop to enable multiple epochs
                # Optionally shuffle the dataframe at the start of each epoch
                if shuffle:
                    np.random.shuffle(array)

                images = []
                labels = []

                for row in array:
                    try:
                        # Read and preprocess image memory stack
                        image_memory_stack = []
                        overflows_starting_timestamp = False
                        for i in range(0, 4):
                            current_index = int(row[3]) - i
                            image_filename = f"{current_index}.png"
                            image_path = os.path.join(image_dir, image_filename)
                            if (
                                not overflows_starting_timestamp
                                and current_index > 0
                                and int(timestamp_array[current_index]) > 0
                                and os.path.exists(image_path)
                            ):
                                image_memory_stack.append(preprocess_image(image_path))
                            else:
                                overflows_starting_timestamp = True
                                image_memory_stack.append(np.zeros((100, 400)))

                        # Combine memory stack and add to list
                        combined_image = np.concatenate(image_memory_stack)
                        images.append(combined_image)
                        labels.append([float(row[0]), float(row[1])])  # Adjust column names as needed

                        # Augmentation: flipped image
                        flipped_memory_stack = np.fliplr(combined_image)
                        images.append(flipped_memory_stack)
                        labels.append([-float(row[0]), float(row[1])])

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
        return returning_generator()

    return data_generator(train_array, True), data_generator(test_array, False)

if __name__ == "__main__":
    # Example of how to use the function
    train_generator, test_generator = prepare_image_data_generator(
		image_dir="D:/bachelor arbeit/reduced_data/images",
		csv_path="D:/bachelor arbeit/reduced_data/data.csv",
        batch_size=200
	)
    model = Sequential([
    # Preprocessing layer
    Lambda(lambda x: x/255, input_shape=(4, 400, 100, 1)),
    
    # Replace Conv2D with ConvLSTM2D
    ConvLSTM2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv_lstm1'),
    BatchNormalization(),
    ConvLSTM2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv_lstm2'),
    BatchNormalization(),
    ConvLSTM2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv_lstm3'),
    ConvLSTM2D(64, kernel_size=(3, 3), activation='relu', name='conv_lstm4'),
    ConvLSTM2D(64, kernel_size=(3, 3), activation='relu', name='conv_lstm5'),
    
    # Flatten temporal features
    Flatten(),
    
    # Dense layers remain similar
    Dense(100,),
    Dense(50),
    Dense(10),
    Dense(2, activation='tanh') # Ensures final output in [-1, 1]
])

    model.compile(loss = 'mse', optimizer = 'adam')

    history= model.fit(x=train_generator,
                       validation_data=test_generator,
                       validation_batch_size=200,
                       validation_steps=13,
                       batch_size=200,
                       steps_per_epoch=55,
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
    outfile = open('./model.json', 'w') 
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