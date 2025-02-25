import json
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from agent.memory_stack import MemoryStack
import agent.utils
from keras.optimizers import AdamW
import math
import keras

MEMORY_STACK_MAX_SIZE = 10

def prepare_image_data_generator(
    image_dir,
    csv_path,
    min_fps,
    max_fps,
    batch_size=32,
    test_split=0.2,
    mirror_images=False,
):
    global MEMORY_STACK_MAX_SIZE
    # 1. Read CSV file
    array = pd.read_csv(csv_path).to_numpy()
    timestamp_array = array[:, 2]
    array = np.column_stack((array, np.arange(len(array))))
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
                        image_memory_stack = np.zeros((MEMORY_STACK_MAX_SIZE, 100, 400))
                        stack_size = 0
                        current_index = int(row[3])
                        next_timestamp = None
                        while stack_size < MEMORY_STACK_MAX_SIZE:
                            image_filename = f"{current_index}.png"
                            image_path = os.path.join(image_dir, image_filename)
                            current_timestamp = int(timestamp_array[current_index])

                            if os.path.exists(image_path) and (
                                next_timestamp == None or
                                (next_timestamp - current_timestamp) > 1000 * 1/16 # 16fps is the frame rate on which the agent is able to operate
                            ):
                                image_memory_stack[:-1] = image_memory_stack[1:]
                                image_memory_stack[-1] = MemoryStack.preprocess(cv2.imread(image_path))
                                stack_size += 1
                                next_timestamp = current_timestamp


                            if current_timestamp == 0:
                                break
                            current_index -= 1
                        image_memory_stack[MEMORY_STACK_MAX_SIZE - stack_size:] = image_memory_stack[MEMORY_STACK_MAX_SIZE - stack_size:][::-1]

                        # Combine memory stack and add to list
                        combined_image = image_memory_stack
                        images.append(combined_image)
                        labels.append([float(row[0]), float(row[1])])  # Adjust column names as needed

                        if mirror_images:
                            # Augmentation: flipped image
                            flipped_memory_stack = np.flip(combined_image, 2)
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
                    yield np.array(images) / 127.5 - 1, np.array(labels)
        return returning_generator()

    return data_generator(train_array, True), data_generator(test_array, False), len(train_array) * (2 if mirror_images else 1), len(test_array) * (2 if mirror_images else 1)

if __name__ == "__main__":
    # Example of how to use the function
    train_generator, test_generator, train_array_length, test_array_length = prepare_image_data_generator(
		image_dir="reduced_data/images",
		csv_path="reduced_data/data.csv",
        batch_size=32,
        mirror_images=True
	)

    model = agent.utils.load_model(None, agent.utils.ModelVersion.LARQV2)

    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.compile(loss = 'mse', optimizer = optimizer)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5",  # Save to this file
        monitor="val_loss",        # Save based on validation loss
        save_best_only=True,       # Only save if the model improves
        save_weights_only=False,   # Save full model (not just weights)
        mode="min",                # Minimize validation loss
        verbose=1
    )
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )

    history= model.fit(x=train_generator,
                       validation_data=test_generator,
                       validation_batch_size=32,
                       validation_steps=math.ceil(test_array_length / 32),
                       batch_size=32,
                       steps_per_epoch=math.ceil(train_array_length / 32),
                       epochs=100, 
                       callbacks=[early_stopping_callback, reduce_lr_callback, checkpoint_callback])
    
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