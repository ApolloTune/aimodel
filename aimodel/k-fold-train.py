import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig



# Initialize BERT tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased", do_lower_case=True)
config = AutoConfig.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=4)
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", config=config)


# Define a custom PyTorch dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=510, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label)}


# Convert dataframe to dataset
df_train = pd.read_csv("../dataset/train.csv", index_col="id", encoding="utf-8")
dataset = TextDataset(df_train['lyrics'].tolist(), df_train['class'].tolist())

# Define k-fold cross-validation
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize lists to store accuracies for each fold
fold_accuracies = []

# Perform k-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(skf.split(df_train['lyrics'], df_train['class'])):
    print(f"Training Fold {fold + 1}/{k_folds}")

    # Split dataset into train and validation sets for the current fold
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model.to(device)
    model.train()
    for epoch in range(3):  # Adjust the number of epochs as needed
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            val_predictions.extend(predicted_labels.tolist())
            val_labels.extend(labels.tolist())

    fold_accuracy = accuracy_score(val_labels, val_predictions)
    fold_accuracies.append(fold_accuracy)
    print(f"Accuracy for Fold {fold + 1}: {fold_accuracy}")

# Calculate average accuracy across all folds
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print(f"Average Accuracy: {average_accuracy}")
torch.save(model.state_dict(), 'training_model_5_fold_lyrics.pth')
