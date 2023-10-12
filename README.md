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

