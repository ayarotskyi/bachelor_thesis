import agent.utils
import keras_tuner as kt
import keras
from data_generator import data_generator, create_tf_dataset
import numpy as np
import pandas as pd


def prepare_datasets(
    image_dir,
    csv_path,
    min_fps,
    max_fps,
    batch_size=32,
    test_split=0.2,
):
    # 1. Read CSV file
    array = pd.read_csv(csv_path).to_numpy()[:64]
    original_array = np.copy(array)
    index_array = np.arange(len(array))

    np.random.shuffle(array)
    split_index = int(len(index_array) * (1 - test_split))
    train_array, test_array = index_array[:split_index], index_array[split_index:]

    def train_dataset(hp: kt.HyperParameters):
        return create_tf_dataset(
            data_generator(
                train_array,
                image_dir,
                original_array,
                10,
                augmentation_multiplier=hp.Int("augmentation_multiplier", 1, 4),
                min_fps=min_fps,
                max_fps=max_fps,
            ),
            batch_size=batch_size,
        )

    def test_dataset(hp: kt.HyperParameters):
        return create_tf_dataset(
            data_generator(
                test_array,
                image_dir,
                original_array,
                10,
                augmentation_multiplier=0,
                min_fps=min_fps,
                max_fps=max_fps,
                shuffle=False,
            ),
            batch_size=batch_size,
        )

    return train_dataset, test_dataset


def build_model(hp: kt.HyperParameters):
    model = agent.utils.load_model(None, hp, agent.utils.ModelVersion.BCNetLSTM)

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_datasets(
        image_dir="reduced_data/images",
        csv_path="reduced_data/data.csv",
        batch_size=32,
        max_fps=6,
        min_fps=4,
    )

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_mae",
        max_trials=50,  # Tests 50 configurations
        executions_per_trial=1,
        directory="kt_model_tuning",
        project_name="model_tuning",
    )

    tuner.search(
        train_dataset,
        validation_data=test_dataset,
        epochs=10,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best augmentation settings found:", best_hps.values)
