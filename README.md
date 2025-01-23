# TODO

Train the model with larq and without larq on these datasets:

1. Using only data for the first difficulty without mirroring
2. Using only data for the first difficulty with mirroring (using updated dataset)
3. Using all the data available without mirroring
4. Using all the data available with mirroring

If non-binary variant is better, test with other edge detectors

# Environment setup:

- MacOS:

1. Install Miniconda
2. Run `conda env create -f tf-metal-arm64.yaml`
3. Run `conda activate tf-metal`
