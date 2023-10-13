
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Train GPT-2 model')
parser.add_argument('--data_path', type=str, default='./framenet_data.pkl', help='Path to FrameNet data')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for AdamW optimizer')
parser.add_argument('--model_dir', type=str, default='./model', help='Directory to save the trained model')
args = parser.parse_args()

# Load FrameNet data
def load_and_process_data(data_path):
    """Load and process FrameNet data.

    Args:
        data_path (str): Path to FrameNet data.

    Returns:
        A pandas DataFrame of the loaded and processed data.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Tokenize data
def tokenize_data(tokenizer, data):
    """Tokenize text data.

    Args:
        tokenizer: Pretrained GPT-2 tokenizer.
        data: DataFrame containing text data.

    Returns:
        A list of tokenized data.
    """
    tokenized_data = []
    for text in data['text']:
        tokenized_text = tokenizer.encode(text)
        tokenized_data.append(tokenized_text)
    return tokenized_data

# Prepare model
def prepare_model(lr, eps):
    """Load GPT-2 model and configure optimizer.

    Args:
        lr (float): Learning rate.
        eps (float): Epsilon for AdamW optimizer.

    Returns:
        model: GPT-2 model.
        optimizer: AdamW optimizer.
    """
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    return model, optimizer

# Perform a single training step
def perform_training_step(model, optimizer, batch, device):
    """Perform a single training step.

    Args:
        model: GPT-2 model.
        optimizer: AdamW optimizer.
        batch: Batch of data for training.
        device: Device to train on.
    """
    model.zero_grad()
    outputs = model(batch, labels=batch)
    loss = outputs[0]
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model over epochs
def train(epochs, model, optimizer, dataloader, device):
    """Train the model over epochs.

    Args:
        epochs (int): Number of epochs.
        model: GPT-2 model.
        optimizer: AdamW optimizer.
        dataloader: DataLoader for the training data.
        device: Device to train on.
    """
    model = model.to(device)
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = perform_training_step(model, optimizer, batch, device)
            total_loss += loss
        print(f'Epoch: {epoch+1}, Loss: {total_loss}')

# Save the trained model
def save_model(model, model_dir):
    """Save the trained model to a directory.

    Args:
        model: Trained GPT-2 model.
        model_dir (str): Directory to save the trained model.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f'Saving model to {model_dir}')
    model.save_pretrained(model_dir)

# Load and process data
if __name__ == '__main__':
    print('Loading and processing data...')
    data = load_and_process_data(args.data_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize the data
    print('Tokenizing data...')
    tokenized_data = tokenize_data(tokenizer, data)

    # Prepare datasets and data loader
    BATCH_SIZE = 8
    data = np.array(tokenized_data, dtype=object)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=2)

    # Prepare model
    print('Preparing model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer = prepare_model(args.lr, args.eps)

    # Train the model
    print(f'Training model on {device} for {args.epochs} epochs...')
    train(args.epochs, model, optimizer, dataloader, device)

    # Save the trained model
    save_model(model, args.model_dir)
    print('Training and saving model process completed.')
