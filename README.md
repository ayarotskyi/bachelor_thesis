# TODO

- Add controls data to the memory stack
- Switch from stacking images to using LSTM layers (or Conv3D in larq)
- Train the model with larq and without larq on these datasets:

1. Using only data for the first difficulty without collisions handling (until row 15010 in the updated dataset)
2. Using only data for the first difficulty with collisions handling (until row 16642)
3. Using all the data available
4. All the previous options with mirorring

If non-binary variant is better, test with other edge detectors

# Environment setup:

- MacOS:

1. Install Miniconda
2. Run `conda env create -f tf-metal-arm64.yaml`
3. Run `conda activate tf-metal`
