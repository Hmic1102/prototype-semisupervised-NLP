import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchtext
import torchdata
from scipy.spatial.distance import euclidean
from scipy.stats import gamma
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW


def get_prototype(data):
    return np.mean(data, axis=0)

def get_distances(data, prototype):
    return np.array([euclidean(d, prototype) for d in data])

def get_gamma_parameters(distances):
    mean = np.mean(distances)
    std = np.std(distances)
    alpha = (mean / std) ** 2
    beta = std ** 2 / mean
    return alpha, beta

def get_class_probabilities(Distances, parameters):
    probabilities = []
    for i in range(len(Distances)):  # Loop over each unlabeled data point
        prob_row = []
        for j in range(len(Distances[0])):  # Loop over each prototype
            alpha, beta = parameters[j]
            distance = Distances[i][j]
            prob = gamma.pdf(distance, a=alpha, scale=beta)
            prob_row.append(prob)
        probabilities.append(prob_row)
    return probabilities
def test_accu(num,model):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  correct_baseline = 0
  total_baseline = 0

  test_dataset = IMDB(".torchtext/cache", split='test')

  test_data = [(label, text) for label, text in test_dataset]
  random.shuffle(test_data)
  test_data_subset = test_data[:num]
  class MyDataset(Dataset):
    def __init__(self, texts, hard_labels, tokenizer):
        self.texts = texts
        self.hard_labels = hard_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        hard_label = torch.tensor(self.hard_labels[idx], dtype=torch.long)

        tokenized_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']

        return {'input_ids': input_ids.squeeze(0), 'attention_mask': attention_mask.squeeze(0), 'hard_label': hard_label}
  test_loader = DataLoader(MyDataset([text for _, text in test_data_subset],
                                    [label - 1 for label, _ in test_data_subset],
                                    tokenizer),
                          batch_size=4)
  with torch.no_grad():
    model.eval()
    for i, batch in enumerate(test_loader):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['hard_label'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask)
      predicted = torch.argmax(outputs.logits, dim=1)
      total_baseline += labels.size(0)
      correct_baseline += (predicted == labels).sum().item()

  baseline_accuracy = correct_baseline / total_baseline
  print(f"finetune Accuracy: {baseline_accuracy:.4f}")

# Set random seed for reproducibility

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Load dataset
test_dataset = IMDB(root=".torchtext/cache", split='test')
test_data = [(label, text) for label, text in test_dataset]
random.shuffle(test_data)
test_data_subset = test_data[:50]

train_dataset = IMDB(".torchtext/cache", split='train')
# Convert dataset to list and then shuffle and select a subset
train_data = [(label, text) for label, text in train_dataset]
random.shuffle(train_data)
train_data_subset = train_data[:20]

# Compute soft labels
with open('IMBD.npy', 'rb') as f:
    proto = np.load(f, allow_pickle=True)

def to_prob(distances, temperature=1.0):
    similarity_scores = -np.array(distances) / temperature
    exp_scores = np.exp(similarity_scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities

Distances = []
hard = []
for label, line in train_data_subset:
    inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask']
    sum_embeddings = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
    mean_embeddings = sum_embeddings / mask.sum(dim=1, keepdim=True)
    Distance = [0] * 2
    for i in proto.flatten()[0]:
        Distance[i-1] = euclidean(mean_embeddings.cpu().flatten(), proto.flatten()[0][i])
    Distances.append(Distance)
    hard.append(label - 1)
Distances =[pd.DataFrame(Distances)[pd.Series(hard)==0][0].to_numpy(),pd.DataFrame(Distances)[pd.Series(hard)==1][1].to_numpy()]
gamma_parameters = [get_gamma_parameters(distance) for distance in Distances]
print(gamma_parameters)

Distances = []
text = []
hard = []
temperature = 0.2
for label, line in test_data_subset:
    inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask']
    sum_embeddings = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
    mean_embeddings = sum_embeddings / mask.sum(dim=1, keepdim=True)
    Distance = [0] * 2
    for i in proto.flatten()[0]:
        Distance[i-1] = euclidean(mean_embeddings.cpu().flatten(), proto.flatten()[0][i])
    Distances.append(Distance)
    text.append(line)
    hard.append(label - 1)
Distances = get_class_probabilities(Distances,gamma_parameters)
Prob = [to_prob(i, temperature) for i in Distances]
labels = [np.argmax(i) for i in Distances]


# Define custom dataset
class MyDataset(Dataset):
    def __init__(self, texts, hard_labels, soft_labels, tokenizer):
        self.texts = texts
        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        hard_label = torch.tensor(self.hard_labels[idx], dtype=torch.long)
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float)
        tokenized_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        return {'input_ids': input_ids.squeeze(0), 'attention_mask': attention_mask.squeeze(0), 'hard_label': hard_label, 'soft_label': soft_label}

# Create dataset and dataloader
train_dataset = MyDataset(text, labels, Prob, tokenizer)
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model, optimizer, and hyperparameters
student = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2).to(device)
optimizer = AdamW(student.parameters(), lr=args.plr)
alpha = args.alpha
temperature = 2.0
epochs = 10

# Train model
for epoch in range(epochs):
    student.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hard_labels = batch['hard_label'].to(device)
        soft_labels = batch['soft_label'].to(device)

        outputs = student(inputs, attention_mask=attention_mask)

        loss_hard = F.cross_entropy(outputs.logits, hard_labels)
        loss_soft = (F.kl_div(
            F.log_softmax(outputs.logits / temperature, dim=-1),
            F.softmax(soft_labels / temperature, dim=-1),
            reduction='batchmean'
        )) * temperature * temperature

        loss = alpha * loss_soft + (1 - alpha) * loss_hard

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")





test_accu(50,student)

class MyDataset(Dataset):
  def __init__(self, texts, hard_labels, tokenizer):
    self.texts = texts
    self.hard_labels = hard_labels
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    hard_label = torch.tensor(self.hard_labels[idx], dtype=torch.long)

    tokenized_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']

    return {'input_ids': input_ids.squeeze(0), 'attention_mask': attention_mask.squeeze(0), 'hard_label': hard_label}

train_dataset = IMDB(".torchtext/cache", split='train')
# Convert dataset to list and then shuffle and select a subset
train_data = [(label, text) for label, text in train_dataset]
random.shuffle(train_data)
train_data_subset = train_data[:20]
train_loader = DataLoader(MyDataset([text for _, text in train_data_subset],
                                    [label - 1 for label, _ in train_data_subset],
                                    tokenizer),
                          batch_size=4)

optimizer = AdamW(student.parameters(), lr=args.flr)
epochs = 10

# Train the model
for epoch in range(epochs):
    student.train()
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hard_labels = batch['hard_label'].to(device)

        outputs = student(input_ids, attention_mask=attention_mask)
        loss_hard = F.cross_entropy(outputs.logits, hard_labels)

        optimizer.zero_grad()
        loss_hard.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss_hard.item()}")
    test_accu(50,student)
