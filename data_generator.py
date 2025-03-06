import numpy as np
import os
import tensorflow as tf
from agent.memory_stack import MemoryStack
import cv2

def data_generator(array, image_dir, timestamp_array, memory_stack_size, min_fps, max_fps, augmentations=None, shuffle=True):
    augmentations = augmentations if augmentations else []

    def generator():
        if shuffle:
            np.random.shuffle(array)

        for row in array:
            try:
                # Create memory stack
                image_memory_stack = np.zeros((memory_stack_size, 100, 400))
                stack_size = 0
                current_index = int(row[3])
                next_timestamp = None

                while stack_size < memory_stack_size:
                    image_filename = f"{current_index}.png"
                    image_path = os.path.join(image_dir, image_filename)
                    current_timestamp = int(timestamp_array[current_index])

                    if os.path.exists(image_path) and (
                        next_timestamp is None or
                        (next_timestamp - current_timestamp) > 1000 * 1/min_fps
                    ):
                        image_memory_stack = \
                            np.concatenate([image_memory_stack[1:], [MemoryStack.preprocess(cv2.imread(image_path))]])
                        stack_size += 1
                        next_timestamp = current_timestamp

                    if current_timestamp == 0:
                        break
                    current_index -= 1
                image_memory_stack[memory_stack_size - stack_size:] = \
                    image_memory_stack[memory_stack_size - stack_size:][::-1]
                image_memory_stack = np.concatenate(image_memory_stack) / 127.5 - 1

                label = [float(row[0]), float(row[1])]

                # Yield original
                yield {
                    "cnn_input": image_memory_stack,
                    "dense_input": row[4:].reshape(10,2)
                }, label

            except Exception as e:
                print(f"Error processing data: {e}")

    return generator

def create_tf_dataset(generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "cnn_input": tf.TensorSpec(shape=(1000, 400), dtype=tf.int8),  # First input (CNN)
                "dense_input": tf.TensorSpec(shape=(10, 2), dtype=tf.float32), # Second input (Dense)
            },
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
        )
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset