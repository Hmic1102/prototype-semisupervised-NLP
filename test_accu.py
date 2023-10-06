import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
import torch.nn.functional as F
from scipy.spatial.distance import euclidean

import torchtext
import torchdata
import random
from transformers import BertTokenizer, BertModel
import torch
from torchtext.datasets import IMDB
import argparse


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
test_data_subset = test_data[:5000]

# Compute soft labels
with open('IMBD.npy', 'rb') as f:
    proto = np.load(f, allow_pickle=True)

def to_prob(distances, temperature=1.0):
    similarity_scores = -np.array(distances) / temperature
    exp_scores = np.exp(similarity_scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities



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

Prob = [to_prob(i, temperature) for i in Distances]
labels = [np.argmax(i) for i in Prob]

def accu(label1, label2):
  count = 0
  for index, i in enumerate(label1):
    if i == label2[index]:
      count+=1
  return count/len(label1)
print(f'accu of distance is {accu(labels,hard)}')





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






train_dataset = IMDB(".torchtext/cache", split='train')
train_data = [(label, text) for label, text in train_dataset]
random.shuffle(train_data)
train_data_subset = train_data[:200]




Distances_gamma = []
hard_gamma = []
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
    Distances_gamma.append(Distance)
    hard_gamma.append(label - 1)
Distances =[pd.DataFrame(Distances_gamma)[pd.Series(hard_gamma)==0][0].to_numpy(),pd.DataFrame(Distances_gamma)[pd.Series(hard_gamma)==1][1].to_numpy()]
gamma_parameters = [get_gamma_parameters(distance) for distance in Distances_gamma]
print(gamma_parameters)


Distances = get_class_probabilities(Distances,gamma_parameters)
Prob = [to_prob(i, temperature) for i in Distances]
labels = [np.argmax(i) for i in Prob]
print(f'accu of gamma_prob is {accu(labels,hard)}')
