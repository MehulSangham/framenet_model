
# FrameNet Model

This repository contains scripts and files for a project aimed to fine-tune a GPT-2 model for the specific task of identifying semantic frames in text data. The input to this process can be any amount of text data in multiple formats, and the output will be the identified semantic frames in the corpus.

## Contents

The repository contains the following key files:

- `data_prep.py`: This Python script is responsible for preparing the text data to be fed into the GPT-2 model.
- `train_gpt2.py`: This Python script handles loading the processed data, preparing it for the GPT-2 model, and fine-tuning the GPT-2 model on the data.
- `Jupyter Notebook`: TBD

## Jupyter Notebook

TBD

## Logging

TBD

## Training Process

TBD

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

echo "## active_line 2 ##"
## load_and_process_framenet.py
echo "## active_line 3 ##"

echo "## active_line 4 ##"
The script `load_and_process_framenet.py` is used to load and process FrameNet data using the nltk.corpus.framenet module. Here is a detailed explanation of the script components and process flow:
echo "## active_line 5 ##"

echo "## active_line 6 ##"
1. **Argument Parsing**: The argparse module is used to manage the script
