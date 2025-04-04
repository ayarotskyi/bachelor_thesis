import keras_tuner as kt
import hypermodel


if __name__ == "__main__":
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel.MyHyperModel(),
        objective="val_loss",
        max_trials=50,  # Tests 50 configurations
        executions_per_trial=1,
        directory="learning_rate_tuning",
        project_name="model_tuning",
        overwrite=False,
    )

    tuner.search(
        epochs=5,
        callbacks=[],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best augmentation settings found:", best_hps.values)
