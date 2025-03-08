import numpy as np
import os
import tensorflow as tf
from agent.memory_stack import MemoryStack
import cv2
import time


def data_generator(
    index_array,
    image_dir,
    original_array,
    memory_stack_size,
    min_fps,
    max_fps,
    augmentations=None,
    shuffle=True,
):
    augmentations = augmentations if augmentations else []

    def generator():
        if shuffle:
            np.random.shuffle(index_array)

        for index in index_array:
            # Create memory stack
            image_memory_stack = np.zeros((memory_stack_size, 100, 200))
            stack_size = 0
            current_index = index
            next_timestamp = None

            while stack_size < memory_stack_size:
                image_filename = f"{current_index}.png"
                image_path = os.path.join(image_dir, image_filename)
                current_timestamp = int(original_array[current_index][2])

                if os.path.exists(image_path) and (
                    next_timestamp is None
                    or (next_timestamp - current_timestamp) > 1000 * 1 / min_fps
                ):
                    image_memory_stack = np.concatenate(
                        [
                            image_memory_stack[1:],
                            [MemoryStack.preprocess(cv2.imread(image_path))],
                        ]
                    )
                    stack_size += 1
                    next_timestamp = current_timestamp

                if current_timestamp == 0:
                    break
                current_index -= 1
            image_memory_stack[memory_stack_size - stack_size :] = image_memory_stack[
                memory_stack_size - stack_size :
            ][::-1]

            label = None
            base_timestamp = int(original_array[index][2])
            current_index = index + 1
            while label is None:
                timestamp = (
                    int(original_array[current_index][2])
                    if current_index < len(original_array)
                    else 0
                )
                if timestamp == 0:
                    label = [
                        float(original_array[current_index - 1][0]),
                        float(original_array[current_index - 1][1]),
                    ]
                if timestamp - base_timestamp > 1000 * 1 / min_fps:
                    label = [
                        float(original_array[current_index][0]),
                        float(original_array[current_index][1]),
                    ]
                current_index += 1

            yield (
                image_memory_stack / 127.5 - 1,
                label,
            )

    return generator


def create_tf_dataset(generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(10, 100, 200), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
        ),
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
