import os
import urllib.request
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
from transformers import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score



# Download the AG News dataset
url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
urllib.request.urlretrieve(url, "ag_news_train.csv")



# Load the AG News dataset into a pandas dataframe
train_df = pd.read_csv("ag_news_train.csv", header=None)
test_df = pd.read_csv("ag_news_test.csv", header=None)

# Combine the title and description columns
train_df[1] = train_df[1] + ' ' + train_df[2]
test_df[1] = test_df[1] + ' ' + test_df[2]

# Drop the description column
train_df = train_df.drop(columns=[2])
test_df = test_df.drop(columns=[2])

# Rename the columns
train_df.columns = ['label', 'text']
test_df.columns = ['label', 'text']

# Split the data into training, validation, and testing sets
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    
    return text

train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Convert the labels into integers
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
val_df['label'] = label_encoder.transform(val_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

num_classes = len(label_encoder.classes_)
print("Number of classes: ", num_classes)
print(train_df.head())
from transformers import BertTokenizer

# Instantiate the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the text data
MAX_LEN=128
MAX_LEN = 128
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Create the data loaders
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

# Instantiate the BERT classifier
model = BertClassifier(num_classes=4).to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Train the model
best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    # Train the model for one epoch
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            val_loss += loss.item()
            val_preds.extend(torch.argmax(logits, axis=1).tolist())
            val_labels.extend(labels.tolist())
    
    # Compute the validation accuracy, precision, recall, and F1-score
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_score,recall_score,f1_score(val_labels, val_preds, average='weighted')
    
    print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_dataloader):.4f}, val_loss={val_loss/len(val_dataloader):.4f}, val_accuracy={val_accuracy:.4f}, val_precision={val_precision:.4f}, val_recall={val_recall:.4f}, val_f1={val_f1:.4f}")
    
    # Save the best performing model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        torch.save(model.state_dict(), 'best_model.pt')
        best_val_accuracy = val_accuracy
# Define function for evaluating the model
def evaluate_model(model, test_data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    print('Test Accuracy:', accuracy)
    print('Test Precision:', precision)
    print('Test Recall:', recall)
    print('Test F1-score:', f1)
# Evaluate the model on the test set
evaluate_model(model, test_data_loader, device)
