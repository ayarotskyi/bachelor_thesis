import json
import keras.callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import agent.utils
import keras
from data_generator import data_generator, create_tf_dataset


def prepare_datasets(
    image_dir, csv_path, min_fps, max_fps, batch_size=32, test_split=0.2, memory_size=8
):
    # 1. Read CSV file
    array = pd.read_csv(csv_path).to_numpy()[:15008]
    original_array = np.copy(array)
    index_array = np.arange(len(array))

    np.random.shuffle(array)
    split_index = int(len(index_array) * (1 - test_split))
    train_array, test_array = index_array[:split_index], index_array[split_index:]

    train_dataset = create_tf_dataset(
        data_generator(
            train_array,
            image_dir,
            original_array,
            memory_size,
            augmentation_multiplier=1,
            min_fps=min_fps,
            max_fps=max_fps,
        ),
        memory_size=memory_size,
        batch_size=batch_size,
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
        memory_size=memory_size,
        batch_size=batch_size,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    memory_size = 8
    # Example of how to use the function
    train_dataset, test_dataset = prepare_datasets(
        image_dir="reduced_data/images",
        csv_path="reduced_data/data.csv",
        batch_size=32,
        max_fps=6,
        min_fps=4,
        memory_size=memory_size,
    )

    model = agent.utils.load_model(
        None, memory_size, agent.utils.ModelVersion.BCNetLSTM
    )
    print(model.summary())

    optimizer = keras.optimizers.Adam(learning_rate=0.00025)

    model.compile(loss="mse", optimizer=optimizer)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5",  # Save to this file
        monitor="val_loss",  # Save based on validation loss
        save_best_only=True,  # Only save if the model improves
        save_weights_only=False,  # Save full model (not just weights)
        mode="min",  # Minimize validation loss
        verbose=1,
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=100,
        callbacks=[early_stopping_callback, reduce_lr_callback, checkpoint_callback],
    )

    model.save("model.h5")
    outfile = open("./model.json", "w")
    json.dump(model.to_json(), outfile)
    outfile.close()
    model.save_weights("model.weights.h5")

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
