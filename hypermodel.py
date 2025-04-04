import keras
import keras_tuner as kt
import pandas as pd
import numpy as np
from data_generator import data_generator, create_tf_dataset
from keras.layers import (
    TimeDistributed,
    Conv2D,
    BatchNormalization,
    Flatten,
    Dropout,
    LSTM,
)
from keras.models import Sequential
from keras.regularizers import l2


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
    original_array = pd.read_csv(csv_path).to_numpy()[:15008]
    index_array = rng.choice(
        np.arange(len(original_array)), size=500, replace=False, axis=0
    )

    np.random.shuffle(index_array)
    split_index = int(len(index_array) * (1 - test_split))
    train_array, test_array = index_array[:split_index], index_array[split_index:]

    memory_size = 8
    target_size = hp.get("target_size")

    train_dataset = create_tf_dataset(
        data_generator(
            train_array,
            image_dir,
            original_array,
            memory_size,
            augmentation_multiplier=3,
            min_fps=min_fps,
            max_fps=max_fps,
            target_size=target_size,
        ),
        batch_size=batch_size,
        memory_size=memory_size,
        target_size=target_size,
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
            target_size=target_size,
        ),
        batch_size=batch_size,
        memory_size=memory_size,
        target_size=target_size,
    )

    return train_dataset, test_dataset


class MyHyperModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters):
        memory_size = 8

        target_size = hp.Int("target_size", 100, 300, 50)

        time_steps = memory_size  # Time domain size
        ch, row, col = 1, int(target_size / 2), target_size  # Updated dimensions

        model = Sequential()

        dropout_rate = 0.2
        regularization_rate = 1e-4

        # Convolutional layers
        model.add(
            TimeDistributed(
                Conv2D(
                    24,
                    kernel_size=(int(target_size / 50), int(target_size / 50)),
                    strides=(2, 2),
                    activation="relu",
                    name="conv1",
                    kernel_regularizer=l2(regularization_rate),
                ),
                input_shape=(time_steps, row, col, ch),
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(
            TimeDistributed(
                Conv2D(
                    36,
                    kernel_size=(int(target_size / 50), int(target_size / 50)),
                    strides=(2, 2),
                    activation="relu",
                    name="conv2",
                    kernel_regularizer=l2(regularization_rate),
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(
            TimeDistributed(
                Conv2D(
                    48,
                    kernel_size=(int(target_size / 50), int(target_size / 50)),
                    strides=(2, 2),
                    activation="relu",
                    name="conv3",
                    kernel_regularizer=l2(regularization_rate),
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(
            TimeDistributed(
                Conv2D(
                    64,
                    kernel_size=(
                        max(int(target_size / 50) - 2, 2),
                        max(int(target_size / 50) - 2, 2),
                    ),
                    activation="relu",
                    name="conv4",
                    kernel_regularizer=l2(regularization_rate),
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(
            TimeDistributed(
                Conv2D(
                    64,
                    kernel_size=(
                        max(int(target_size / 50) - 2, 2),
                        max(int(target_size / 50) - 2, 2),
                    ),
                    activation="relu",
                    name="conv5",
                    kernel_regularizer=l2(regularization_rate),
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dropout(dropout_rate + 0.2)))

        # LSTM layers
        model.add(
            LSTM(
                int(target_size / 2),
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=l2(regularization_rate),
            )
        )
        model.add(TimeDistributed(Dropout(dropout_rate + 0.1)))
        model.add(
            LSTM(
                int(target_size / 4),
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=l2(regularization_rate),
            )
        )
        model.add(TimeDistributed(Dropout(dropout_rate + 0.2)))
        third_lstm_layer = hp.Boolean("third_lstm_layer")
        if third_lstm_layer:
            model.add(
                LSTM(
                    int(target_size / 20),
                    activation="tanh",
                    return_sequences=True,
                    kernel_regularizer=l2(regularization_rate),
                )
            )
            model.add(TimeDistributed(Dropout(dropout_rate + 0.1)))
        model.add(
            LSTM(
                2,
                activation="tanh",
                return_sequences=False,
                kernel_regularizer=l2(regularization_rate),
            )
        )
        model.compile(
            optimizer=keras.optimizers.Adam(0.00025),
            loss="mse",
            metrics=["mae"],
        )
        model.summary()
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
