
import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
import logging

# Initialize the logger
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define the command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='/Users/mehulsangham/GH Projects/framenet_model/processed_framenet_data.pkl', help='Path to the processed FrameNet data.')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for ADAM.')
parser.add_argument('--eps', type=float, default=0.00001, help='Epsilon for ADAM.')
args = parser.parse_args()

# FrameNet Data Loader
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item]

# Function to load and process data
def load_and_process_data(data_path: str) -> pd.DataFrame:
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    return data

# Tokenize data
def tokenize_data(tokenizer, data: pd.DataFrame):
    return [tokenizer.encode(x, add_special_tokens=True) for x in data["definition"].tolist()]

# Train the GPT-2 model
def train(model, tokenized_data, device, epochs, lr, eps):
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    for epoch in range(epochs):
        logging.info(f'Starting epoch {epoch + 1}/{epochs}.')
        # Data loader
        train_dataloader = DataLoader(TextDataset(tokenized_data), batch_size=1, shuffle=True)
        for step, batch in enumerate(train_dataloader):
            inputs = torch.tensor(batch).unsqueeze(0).to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs[0]
            loss = loss / gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        logging.info(f'Finished epoch {epoch + 1}/{epochs}. Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), "fine_tuned_gpt2.pt")
    logging.info('Model saved')

# Main function
if __name__ == "__main__":
    logging.info('Running GPT-2 training script')

    data = load_and_process_data(args.data_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenize_data(tokenizer, data)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Training model on {device}')

    train(model, tokenized_data, device, args.epochs, args.lr, args.eps)

    logging.info('Training completed')
