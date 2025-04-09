# TODO

Train new model with:
(choose which model to use from one of the bottom ones)

1. Image normalization according to the statistic in training data
2. More augmentation (x5)

Test these:

- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/reduced_dimensions (a9d95660104622e08ef5cb2b6b8dc287f6dbc5c9)
- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input (2ab06af424d88276628d1f91f97cf16d062b6d62)
- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input/speed_prediction (b0b0a98a1dabcdfed6926e8787cb23ce70398253)
- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/reduced_dimensions/speed_prediction (3ad2573f3047d7260fa750802d1c7df6b23ec9ba)
- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/reduced_dimensions/speed_prediction/without_restoring_best_weights

# Environment setup:

- MacOS:

1. Install Miniconda
2. Run `conda env create -f tf-metal-arm64.yaml`
3. Run `conda activate tf-metal`
