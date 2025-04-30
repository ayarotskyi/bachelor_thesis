# Best models:

- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input (2ab06af424d88276628d1f91f97cf16d062b6d62)

# Todo:

Train raw input model using:

- std normalization
- whole dataset
- transfer learning on second difficulty

Test these models:

- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input/std_norm
- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input/transfer_learning (image / 127.5 - 1)
- https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input/whole_dataset (image / 127.5 - 1)

# Environment setup:

- MacOS:

1. Install Miniconda
2. Run `conda env create -f tf-metal-arm64.yaml`
3. Run `conda activate tf-metal`
