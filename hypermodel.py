import keras
import keras_tuner as kt
import agent.utils
import pandas as pd
import numpy as np
from data_generator import data_generator, create_tf_dataset


def prepare_datasets(
    image_dir,
    csv_path,
    min_fps,
    max_fps,
    hp: kt.HyperParameters,
    batch_size=32,
    test_split=0.2,
):
    # 1. Read CSV file
    rng = np.random.default_rng()
    array = rng.choice(
        pd.read_csv(csv_path).to_numpy()[:15008], size=500, replace=False, axis=0
    )
    original_array = np.copy(array)
    index_array = np.arange(len(array))

    np.random.shuffle(array)
    split_index = int(len(index_array) * (1 - test_split))
    train_array, test_array = index_array[:split_index], index_array[split_index:]

    memory_size = hp.get("memory_size")

    train_dataset = create_tf_dataset(
        data_generator(
            train_array,
            image_dir,
            original_array,
            memory_size,
            augmentation_multiplier=hp.Int("augmentation_multiplier", 1, 4),
            min_fps=min_fps,
            max_fps=max_fps,
            hp=hp,
        ),
        batch_size=batch_size,
        memory_size=memory_size,
    )

    test_dataset = create_tf_dataset(
        data_generator(
            test_array,
            image_dir,
            original_array,
            memory_size,
            augmentation_multiplier=0,
            min_fps=min_fps,
            max_fps=max_fps,
            shuffle=False,
        ),
        batch_size=batch_size,
        memory_size=memory_size,
    )

    return train_dataset, test_dataset


class MyHyperModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters):
        memory_size = hp.Int("memory_size", 6, 12)
        model = agent.utils.load_model(
            None, hp, memory_size, agent.utils.ModelVersion.BCNetLSTM
        )
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def fit(self, hp, model, **kwargs):
        train_dataset, test_dataset = prepare_datasets(
            image_dir="reduced_data/images",
            csv_path="reduced_data/data.csv",
            batch_size=32,
            max_fps=6,
            min_fps=4,
            hp=hp,
        )
        return model.fit(
            train_dataset,
            validation_data=test_dataset,
            **kwargs,
        )
