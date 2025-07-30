# Use of Imitation Learning in an Obstacle Course Scenario

This repository contains architecture and training pipeline for the CNN capable of autonomously driving NVIDIA's Jetbot through an abstacle course.

## The trained model ([link to model](https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input)) making its way through the course
![Top-down view of the Jetbot being controlled by the CNN](./top-down.gif)

## Visualization of layer activations (old version of the model)
![](./activations_visualization.gif)

# Environment setup:

- MacOS:

1. Install Miniconda
2. Run `conda env create -f tf-metal-arm64.yaml`
3. Run `conda activate tf-metal`

- Windows:

1. Install Miniconda
2. Open any terminal with administrator rights
3. Run `conda env create -f tf-windows.yaml`
4. Run `conda activate tf`

# Script execution:

## Training

1. Download and unzip the reduced_data directory from https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/datasets/all_difficulties_cleaned into the root of the project
2. Run `python training_pipeline.py`
3. When the training is finished the model.h5 file will be created in the root of the project

## Teleoperation (for data collection)

i. If you want to test both server and client on your machine

1. Don't change anything and run `python teleoperation_server.py` in one terminal
2. Open another terminal and run `python teleoperation_client.py`
3. The gamepad data will be randomly generated
4. Wherever you're done just press ctrl+c in both terminals and wait until the data is transferred (optional)

ii. Data acquisition with real JetBot:

1. Obtain the JetBot's address in current network and change the `HOST` variable in both `teleoperation_client.py` and `teleoperation_server.py` files
2. Change `MOCK_GAMEPAD` to `False` in `teleoperation_client.py`
3. Change `MOCK_SERVER` to `False` in `teleoperation_server.py`
4. Upload the project's directory to the JetBot (Optional, since it should already be uploaded in `/test/teleoperation` directory)
5. Connect a gamepad to your machine
6. Run `python teleoperation_server.py` on JetBot (using the `{{JetBot's local ip}}:8888/lab` endpoint [password is `jetbot`])
7. Run `python teleoperation_client.py` on your machine
8. Start controlling the JetBot with your gamepad
9. Stop both scripts using ctrl+c and wait until the data is transferred to your machine

## Agent (control inference pipeline)

1. Obtain the JetBot's address in current network and change the `HOST` variable in both `agent/client.py` and `agent/server.py` files
2. Install the https://github.com/NVIDIA-AI-IOT/jetcam library on JetBot (Optional, since it should already be installed)
3. Upload `agent` directory to JetBot (Optional, since it should already be uploaded under `/test/teleoperation/agent`)
4. Download `model.h5` from https://huggingface.co/datasets/ayarotskyi/bachelor_thesis/tree/main/models/BSNetLSTM/raw_input and upload it to the root of the project on JetBot (eventually should be `test/teleoperation/model.h5`)
5. Run `python agent/server.py` on your machine
6. Run `python agent/client.py` on JetBot
