import numpy as np
import os
import tensorflow as tf
from agent.memory_stack import MemoryStack
import cv2
import random
import keras_tuner as kt


def apply_augmentations(image, hp: kt.HyperParameters):
    if random.randint(0, 100) > 50:
        max_shift = hp.Int("max_shift", 1, 10)
        dx = np.random.randint(-max_shift, max_shift)
        dy = np.random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    if random.randint(0, 100) > 50:
        max_angle = hp.Int("max_angle", 1, 10)
        max_scale_factor = hp.Float("max_scale_factor", 0.01, 0.2)
        angle = np.random.uniform(-max_angle, max_angle)
        scale_factor = np.random.uniform(1 - max_scale_factor, 1 + max_scale_factor)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale_factor)
        image = cv2.warpAffine(image, M, (w, h))

    if random.randint(0, 100) > 50:
        max_sigma = hp.Float("max_sigma", 0.1, 3)
        sigma = np.random.uniform(0, max_sigma)
        noise = np.random.normal(0, sigma, image.shape).astype(
            np.uint8
        )  # Gaussian noise with mean 0 and std 10
        image = cv2.add(image, noise)
        image = np.clip(image, 0, 255)  # Ensure pixel values stay in valid range

    return image


def get_dataset_pair(
    memory_stack_size,
    image_dir,
    original_array,
    index,
    target_fps,
    hp: kt.HyperParameters = None,
):
    # Create memory stack
    image_memory_stack = np.zeros((memory_stack_size, 100, 200))
    stack_size = 0
    current_index = index
    next_timestamp = None

    while stack_size < memory_stack_size and current_index >= 0:
        image_filename = f"{current_index}.png"
        image_path = os.path.join(image_dir, image_filename)
        current_timestamp = int(original_array[current_index][2])

        if os.path.exists(image_path) and (
            next_timestamp is None
            or (next_timestamp - current_timestamp) > 1000 * 1 / target_fps
        ):
            image_memory_stack = np.concatenate(
                [
                    image_memory_stack[1:],
                    [
                        apply_augmentations(
                            MemoryStack.preprocess(cv2.imread(image_path)), hp
                        )
                        if hp is not None
                        else MemoryStack.preprocess(cv2.imread(image_path))
                    ],
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
        if timestamp - base_timestamp > 1000 * 1 / target_fps:
            label = [
                float(original_array[current_index][0]),
                float(original_array[current_index][1]),
            ]
        current_index += 1

    return (
        image_memory_stack / 127.5 - 1,
        label,
    )


def data_generator(
    index_array,
    image_dir,
    original_array,
    memory_stack_size,
    min_fps,
    max_fps,
    augmentation_multiplier,
    shuffle=True,
    hp=None,
):
    def generator():
        if shuffle:
            np.random.shuffle(index_array)
        augmentation_queue = np.array([], dtype=np.int32)

        for index in index_array:
            if augmentation_multiplier > 0:
                augmentation_queue = np.append(
                    augmentation_queue, [index] * augmentation_multiplier
                )
                np.random.shuffle(augmentation_queue)

            if len(augmentation_queue) > 0 and random.randint(0, 1) > 0:
                augmentation_index = augmentation_queue[-1]
                augmentation_queue = augmentation_queue[:-1]

                yield get_dataset_pair(
                    memory_stack_size=memory_stack_size,
                    image_dir=image_dir,
                    original_array=original_array,
                    index=augmentation_index,
                    target_fps=np.random.uniform(
                        min_fps, max_fps
                    ),  # using different frequencies in augmented data
                    hp=hp,
                )

            yield get_dataset_pair(
                memory_stack_size=memory_stack_size,
                image_dir=image_dir,
                original_array=original_array,
                index=index,
                target_fps=min_fps,
            )

        for augmentation_index in augmentation_queue:
            yield get_dataset_pair(
                memory_stack_size=memory_stack_size,
                image_dir=image_dir,
                original_array=original_array,
                index=augmentation_index,
                target_fps=np.random.uniform(min_fps, max_fps),
                hp=hp,
            )

    return generator


def create_tf_dataset(generator, batch_size, memory_size):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(memory_size, 100, 200), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
        ),
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
