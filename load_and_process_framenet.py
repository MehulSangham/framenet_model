
import os
import pandas as pd
from nltk.corpus import framenet as fn
import pickle

def frame_to_dict(frame):
    return {'id': frame.ID, 'name': frame.name, 'definition': frame.definition}

fn_data_path = '/users/mehulsangham/nltk_data/corpora/framenet_v17'
processed_data_path = '/Users/mehulsangham/GH Projects/framenet_model/processed_framenet_data.pkl'

if os.path.exists(fn_data_path):
    print('FrameNet data is found.')
    fn_frames = fn.frames()
    frames_list = [frame_to_dict(frame) for frame in fn_frames]
    df = pd.DataFrame(frames_list)
    with open(processed_data_path, 'wb') as f:
        pickle.dump(df, f)
    print('FrameNet data is processed and saved.')
else:
    print('FrameNet data is not found.')
