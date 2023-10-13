
import os
# Argument parsing
import argparse
parser = argparse.ArgumentParser(description='Train GPT-2 for Semantic Frame Induction')
parser.add_argument('--data_path', type=str, default='processed_framenet_data.pkl', help='Path to pickled FrameNet data')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for Adam optimizer')
parser.add_argument('--model_path', type=str, default='gpt2_trained.pth', help='Path to save the trained model')
args = parser.parse_args()

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

data_path = '/Users/mehulsangham/GH Projects/framenet_model/processed_framenet_data.pkl'
epochs = 1
lr = 0.00001
eps = 0.00001
args = argparse.Namespace()

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
def initialize_model_optimizer(model, lr, eps, device):
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    return model, optimizer

def perform_training_step(model, optimizer, batch, device):
    inputs = torch.tensor(batch).unsqueeze(0).to(device)
    outputs = model(inputs, labels=inputs)
    loss = outputs[0]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def train(epoch, model, optimizer, dataloader, device):
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')
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
        print(f'Finished epoch {epoch + 1}/{epochs}. Loss: {loss.item()}')

    def save_model(model, path):
        torch.save(model.state_dict(), path)
    print('Model saved')

# Main function
if __name__ == "__main__":
    print('Running GPT-2 training script')

    data = load_and_process_data(data_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenize_data(tokenizer, data)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training model on {device}')

    train(model, tokenized_data, device, epochs, lr, eps)

    print('Training completed')
