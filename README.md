# FrameNet Fine-Tuning for GPT-2
echo "## active_line 2 ##"

echo "## active_line 3 ##"
This repository contains scripts for fine-tuning the GPT-2 model on FrameNet data.
echo "## active_line 4 ##"

echo "## active_line 5 ##"
## Description
echo "## active_line 6 ##"

echo "## active_line 7 ##"
The scripts provided in this project are designed to preprocess FrameNet data and fine-tune a GPT-2 model using this data. The main goal is to provide a starting point for training a GPT-2 model to generate text based on FrameNet linguistic frames.
echo "## active_line 8 ##"

echo "## active_line 9 ##"
## Repository Structure
echo "## active_line 10 ##"

echo "## active_line 11 ##"
- `processed_framenet_data.pkl`: This file contains the preprocessed FrameNet data.
echo "## active_line 12 ##"
- `train_gpt2.py`: This Python script handles loading the processed data, preparing it for the GPT-2 model, and fine-tuning the GPT-2 model on the data.
echo "## active_line 13 ##"

echo "## active_line 14 ##"
## Requirements
echo "## active_line 15 ##"

echo "## active_line 16 ##"
Before running the scripts, install the necessary Python libraries included in the `requirements.txt` file using pip:
echo "## active_line 17 ##"

echo "## active_line 18 ##"
```
echo "## active_line 19 ##"
pip install -r requirements.txt
echo "## active_line 20 ##"
```
echo "## active_line 21 ##"

echo "## active_line 22 ##"
## Usage
echo "## active_line 23 ##"

echo "## active_line 24 ##"
To run the GPT-2 training script after installing the requirements, use the following command:
echo "## active_line 25 ##"

echo "## active_line 26 ##"
```
echo "## active_line 27 ##"
python train_gpt2.py
echo "## active_line 28 ##"
```
echo "## active_line 29 ##"

echo "## active_line 30 ##"
This will start the process of loading the data, tokenizing it, and fine-tuning the GPT-2 model on it. The fine-tuned model will be saved as `fine_tuned_gpt2.pt`.
echo "## active_line 31 ##"

echo "## active_line 32 ##"
Please note that fine-tuning GPT-2 is a resource-intensive task that requires a machine with a good amount of computation power, ideally with a strong GPU.
echo "## active_line 33 ##"

