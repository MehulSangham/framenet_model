
import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Load and process FrameNet data')
parser.add_argument('--fn_data_path', type=str, default='./nltk_data/corpora/framenet_v17', help='Path to FrameNet data')
parser.add_argument('--processed_data_path', type=str, default='processed_framenet_data.pkl', help='Path to save processed FrameNet data')
args = parser.parse_args()

import pandas as pd
from nltk.corpus import framenet as fn
import pickle
import nltk

# Function to check if preprocessed file exists
def check_preprocessed_file_exists(file_path):
   """Check if file exists at the given file_path.
   
   Args:
        file_path (str): Path of the file to check.
   
   Returns:
        bool: True if file exists, False otherwise.
   """
     return os.path.exists(file_path)

def frame_to_dict(frame):
    """Converts a framenet frame to a dictionary.
    
    Args:
        frame: The frame from nltk.corpus.framenet.frames.
    
    Returns:
        dict: dictionary of frame data.
    """
    return {'id': frame.ID, 'name': frame.name, 'definition': frame.definition}

# Load and process FrameNet data
if not check_preprocessed_file_exists(args.processed_data_path):
    # Check if FrameNet data exists
    if os.path.exists(args.fn_data_path):
        print('FrameNet data is found.')
    else:
        print('FrameNet data is not found. Downloading...')
        nltk.download('framenet_v17')
        print('FrameNet data downloaded.')
    
    # Load FrameNet frames
    fn_frames = fn.frames()
    
    # Convert each frame to a dictionary and store in a list
    frames_list = [frame_to_dict(frame) for frame in fn_frames]
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(frames_list)
    
    # Save the DataFrame to a pickle file
    with open(args.processed_data_path, 'wb') as f:
        pickle.dump(df, f)
        
    print('FrameNet data is processed and saved.')
else:
    print('Preprocessed FrameNet data already exists.')
