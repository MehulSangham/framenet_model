
import os
import pandas as pd
from nltk.corpus import framenet as fn
import pickle
import nltk

# Function to convert FrameNet frame to dictionary
def frame_to_dict(frame):
    return {'id': frame.ID, 'name': frame.name, 'definition': frame.definition}

# Specify the path to FrameNet data
fn_data_path = '/users/mehulsangham/nltk_data/corpora/framenet_v17'
# Specify the path to save processed data
processed_data_path = '/Users/mehulsangham/GH Projects/framenet_model/processed_framenet_data.pkl'

# Check if FrameNet data exists
if os.path.exists(fn_data_path):
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
with open(processed_data_path, 'wb') as f:
    pickle.dump(df, f)
print('FrameNet data is processed and saved.')
