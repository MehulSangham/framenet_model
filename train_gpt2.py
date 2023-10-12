
echo "## active_line 2 ##"
import torch
echo "## active_line 3 ##"
import transformers
echo "## active_line 4 ##"
import pandas as pd
echo "## active_line 5 ##"
import pickle
echo "## active_line 6 ##"
import numpy as np
echo "## active_line 7 ##"
from torch.utils.data import Dataset, DataLoader
echo "## active_line 8 ##"
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
echo "## active_line 9 ##"

echo "## active_line 10 ##"

echo "## active_line 11 ##"
class TextDataset(Dataset):
echo "## active_line 12 ##"

echo "## active_line 13 ##"
    def __init__(self, texts):
echo "## active_line 14 ##"
        self.texts = texts
echo "## active_line 15 ##"

echo "## active_line 16 ##"
    def __len__(self):
echo "## active_line 17 ##"
        return len(self.texts)
echo "## active_line 18 ##"

echo "## active_line 19 ##"
    def __getitem__(self, item):
echo "## active_line 20 ##"
        return self.texts[item]
echo "## active_line 21 ##"

echo "## active_line 22 ##"

echo "## active_line 23 ##"
def load_and_process_data(data_path: str) -> pd.DataFrame:
echo "## active_line 24 ##"
    with open(data_path, "rb") as file:
echo "## active_line 25 ##"
        data = pickle.load(file)
echo "## active_line 26 ##"
    return data
echo "## active_line 27 ##"

echo "## active_line 28 ##"

echo "## active_line 29 ##"
def tokenize_data(tokenizer, data: pd.DataFrame):
echo "## active_line 30 ##"
    return [tokenizer.encode(x, add_special_tokens=True) for x in data["definition"].tolist()]
echo "## active_line 31 ##"

echo "## active_line 32 ##"

echo "## active_line 33 ##"
def train(model, tokenized_data, device):
echo "## active_line 34 ##"
    model = model.to(device)
echo "## active_line 35 ##"
    model.train()
echo "## active_line 36 ##"

echo "## active_line 37 ##"
    optimizer = AdamW(model.parameters(), lr=1e-5)
echo "## active_line 38 ##"
    num_train_epochs = 1
echo "## active_line 39 ##"
    gradient_accumulation_steps = 1
echo "## active_line 40 ##"
    max_grad_norm = 1.0
echo "## active_line 41 ##"

echo "## active_line 42 ##"
    for epoch in range(num_train_epochs):
echo "## active_line 43 ##"
        print(f"Starting epoch {epoch + 1}/{num_train_epochs}")
echo "## active_line 44 ##"
        # Loop through each batch in our data
echo "## active_line 45 ##"
        train_dataloader = DataLoader(TextDataset(tokenized_data), batch_size=1, shuffle=True)
echo "## active_line 46 ##"
        for step, batch in enumerate(train_dataloader):
echo "## active_line 47 ##"
            inputs = torch.tensor(batch).unsqueeze(0).to(device)
echo "## active_line 48 ##"
            outputs = model(inputs, labels=inputs)
echo "## active_line 49 ##"
            loss = outputs[0]
echo "## active_line 50 ##"
            loss = loss / gradient_accumulation_steps
echo "## active_line 51 ##"
            loss.backward()
echo "## active_line 52 ##"
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
echo "## active_line 53 ##"
            optimizer.step()
echo "## active_line 54 ##"
            optimizer.zero_grad()
echo "## active_line 55 ##"
        print(f"Finished epoch {epoch + 1}/{num_train_epochs}")
echo "## active_line 56 ##"

echo "## active_line 57 ##"
    # Save the fine-tuned model
echo "## active_line 58 ##"
    torch.save(model.state_dict(), "fine_tuned_gpt2.pt")
echo "## active_line 59 ##"
    print("Model saved")
echo "## active_line 60 ##"

echo "## active_line 61 ##"

echo "## active_line 62 ##"
if __name__ == "__main__":
echo "## active_line 63 ##"
    print("Running GPT-2 training script")
echo "## active_line 64 ##"

echo "## active_line 65 ##"
    data_path = "/Users/mehulsangham/GH Projects/framenet_model/processed_framenet_data.pkl"
echo "## active_line 66 ##"
    data = load_and_process_data(data_path)
echo "## active_line 67 ##"

echo "## active_line 68 ##"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
echo "## active_line 69 ##"
    tokenized_data = tokenize_data(tokenizer, data)
echo "## active_line 70 ##"

echo "## active_line 71 ##"
    model = GPT2LMHeadModel.from_pretrained("gpt2")
echo "## active_line 72 ##"

echo "## active_line 73 ##"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
echo "## active_line 74 ##"
    print(f"Training model on {device}")
echo "## active_line 75 ##"

echo "## active_line 76 ##"
    train(model, tokenized_data, device)
echo "## active_line 77 ##"

echo "## active_line 78 ##"
    print("Training completed")
