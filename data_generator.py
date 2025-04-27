import numpy as np
import os
import tensorflow as tf
from agent.memory_stack import MemoryStack
import cv2
import random
from agent.utils import calculate_motor_speeds


def apply_augmentations(image):
    if random.randint(0, 100) > 50:
        max_sigma = 0.5
        sigma = np.random.uniform(0, max_sigma)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
        image = cv2.add(image, noise)
        image = np.clip(image, -1, 1)  # Ensure pixel values stay in valid range

    if random.randint(0, 100) > 50:
        max_shift = 9
        dx = np.random.randint(-max_shift, max_shift)
        dy = np.random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    if random.randint(0, 100) > 50:
        max_angle = 5
        max_scale_factor = 0.06
        angle = np.random.uniform(-max_angle, max_angle)
        scale_factor = np.random.uniform(1 - max_scale_factor, 1 + max_scale_factor)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale_factor)
        image = cv2.warpAffine(image, M, (w, h))

    return image


def get_dataset_pair(
    memory_stack_size,
    image_dir,
    original_array,
    index,
    target_fps,
    augment=False,
    target_size=200,
):
    # Create memory stack
    image_memory_stack = np.zeros(
        (
            memory_stack_size,
            int(target_size / 2),
            int(target_size),
        )
    )
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
                            MemoryStack.preprocess(
                                cv2.imread(image_path), target_size=target_size
                            )
                        )
                        if augment
                        else MemoryStack.preprocess(
                            cv2.imread(image_path), target_size=target_size
                        )
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
        image_memory_stack,
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
    checkpoints: np.ndarray = None,
    target_size=200,
):
    def generator():
        if shuffle:
            np.random.shuffle(index_array)
        augmentation_queue = np.array([], dtype=np.int32)

        for index in index_array:
            passes_checkpoint = True
            if checkpoints is not None:
                closest_checkpoint = checkpoints[0]
                for checkpoint_index in reversed(range(0, len(checkpoints))):
                    checkpoint = checkpoints[checkpoint_index]
                    if checkpoint < index:
                        closest_checkpoint = checkpoint
                        break
                for i in range(closest_checkpoint, index):
                    if float(original_array[i, 2]) == 0:
                        passes_checkpoint = False
                        break
            if not passes_checkpoint:
                continue

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
                    augment=True,
                    target_size=target_size,
                )

            yield get_dataset_pair(
                memory_stack_size=memory_stack_size,
                image_dir=image_dir,
                original_array=original_array,
                index=index,
                target_fps=min_fps,
                target_size=target_size,
            )

        for augmentation_index in augmentation_queue:
            yield get_dataset_pair(
                memory_stack_size=memory_stack_size,
                image_dir=image_dir,
                original_array=original_array,
                index=augmentation_index,
                target_fps=np.random.uniform(min_fps, max_fps),
                augment=True,
                target_size=target_size,
            )

    return generator


def create_tf_dataset(generator, batch_size, memory_size, target_size=200):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(memory_size, int(target_size / 2), target_size), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
        ),
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
