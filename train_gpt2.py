
import torch
import transformers
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW


class TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item]


def load_and_process_data(data_path: str) -> pd.DataFrame:
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    return data


def tokenize_data(tokenizer, data: pd.DataFrame):
    return [tokenizer.encode(x, add_special_tokens=True) for x in data["definition"].tolist()]


def train(model, tokenized_data, device):
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_train_epochs = 1
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    for epoch in range(num_train_epochs):
        print(f"Starting epoch {epoch + 1}/{num_train_epochs}")
        # Loop through each batch in our data
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
        print(f"Finished epoch {epoch + 1}/{num_train_epochs}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_gpt2.pt")
    print("Model saved")


if __name__ == "__main__":
    print("Running GPT-2 training script")

    data_path = "/Users/mehulsangham/GH Projects/framenet_model/processed_framenet_data.pkl"
    data = load_and_process_data(data_path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenize_data(tokenizer, data)

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model on {device}")

    train(model, tokenized_data, device)

    print("Training completed")
