import keras_tuner as kt
import hypermodel


if __name__ == "__main__":
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel.MyHyperModel(),
        objective="val_mae",
        max_trials=50,  # Tests 50 configurations
        executions_per_trial=1,
        directory="kt_model_tuning",
        project_name="model_tuning",
    )

    tuner.search(
        epochs=10,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best augmentation settings found:", best_hps.values)
