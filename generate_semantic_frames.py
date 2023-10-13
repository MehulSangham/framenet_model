
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd

# Load the trained model
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Process new text input
def process_input(text, tokenizer):
    return tokenizer.encode(text, return_tensors='pt')

# Apply the model to the tokenized input
def apply_model(input, model):
    with torch.no_grad():
        outputs = model.generate(input, max_length=150, num_return_sequences=5)
    return outputs

# Collect semantic frames
# This function strongly depends on the nature of the model's output
def collect_frames(outputs, tokenizer):
    frames = []
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        
        # Add some logic here to turn the decoded output into frame name and frame description
        # Let's say we just add the whole output as a frame name with no description
        frame_name = decoded_output
        frame_description = ''
        frames.append({'Name': frame_name, 'Description': frame_description})
    return frames

# Display the frames in a table format
def display_as_table(frames):
    df = pd.DataFrame(frames)
    return df

# Replace with actual text and model path
if __name__ == '__main__':
    text = 'Your input text'
    model_path = 'path/to/your/model.pth'
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = load_model(model_path)
    processed_input = process_input(text, tokenizer)
    model_outputs = apply_model(processed_input, model)
    frames = collect_frames(model_outputs, tokenizer)
    df = display_as_table(frames)
    print(df)
