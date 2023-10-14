# FrameNet Processing and GPT-2 Training
This notebook guides through processing FrameNet data and training a GPT-2 model based on the data.
We programmatically execute the scripts: process_framenet.py, load_and_process_framenet.py, and train_gpt2.py.
Each script is explained before execution.
Ensure all dependencies are installed before executing this notebook.

!pip install torch transformers numpy pandas

# Processing FrameNet Xml Data
The process_framenet.py script parses FrameNet xml data and processes it into a Python pickle object.

!python process_framenet.py --input /path/to/input --output /path/to/output

# Loading and Processing FrameNet Data
The load_and_process_framenet.py script loads the FrameNet data from the created pickle file and processes it.

!python load_and_process_framenet.py --data_path /path/to/data

# Training GPT-2 Model
The train_gpt2.py script utilizes the processed FrameNet data to train a GPT-2 Language Model.

!python train_gpt2.py --data_path /path/to/data --epochs 10 --lr 0.01 --eps 1e-8 --model_dir /path/to/save/model