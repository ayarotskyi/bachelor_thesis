import keras_tuner as kt
import hypermodel
import keras


if __name__ == "__main__":
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel.MyHyperModel(),
        objective="val_mae",
        max_trials=50,  # Tests 50 configurations
        executions_per_trial=1,
        directory="learning_rate_tuning",
        project_name="model_tuning",
        overwrite=False,
    )
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )

    tuner.search(
        epochs=100,
        callbacks=[early_stopping_callback],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best augmentation settings found:", best_hps.values)
