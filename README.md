# FrameNet Fine-Tuning for GPT-2

This repository contains scripts for fine-tuning the GPT-2 model on FrameNet data.

## Description

The scripts provided in this project are designed to preprocess FrameNet data and fine-tune a GPT-2 model using this data. The main goal is to provide a starting point for training a GPT-2 model to generate text based on FrameNet linguistic frames.

## Repository Structure

- `processed_framenet_data.pkl`: This file contains the preprocessed FrameNet data.
- `train_gpt2.py`: This Python script handles loading the processed data, preparing it for the GPT-2 model, and fine-tuning the GPT-2 model on the data.

## Requirements

Before running the scripts, install the necessary Python libraries included in the `requirements.txt` file using pip:

```
pip install -r requirements.txt
```

## Usage

To run the GPT-2 training script after installing the requirements, use the following command:

```
python train_gpt2.py
```

This will start the process of loading the data, tokenizing it, and fine-tuning the GPT-2 model on it. The fine-tuned model will be saved as `fine_tuned_gpt2.pt`.

Please note that fine-tuning GPT-2 is a resource-intensive task that requires a machine with a good amount of computation power, ideally with a strong GPU.



## Using GitHub Codespaces and VS Code Remote - Containers

This repository includes a `.devcontainer` directory that specifies a Python development environment with necessary libraries and Jupyter notebook server, ideally for use with GitHub Codespaces or Visual Studio Code Remote - Containers extension.

To set this up:

1. Push the `.devcontainer` directory to your GitHub repository.
2. Open the repository in a new GitHub Codespace by clicking the 'Code' button on your repository page, then 'Open with Codespaces' and 'New Codespace', or use the Remote - Containers extension in VS Code.
3. Github will build the codespace using the Dockerfile in the `.devcontainer` directory. It should start a new session in your browser with VS Code running in the cloud, but with the code in your repository.

With the provided Dockerfile, a Jupyter notebook server starts when the Docker container is run. To start it manually, run:

```
jupyter notebook
```

Additionally, remember to rebuild the codespace whenever you make changes to the Dockerfile or devcontainer.json. You can do this through the Command Palette (F1 or CMD + Shift + P) by typing 'Rebuild Container'.
