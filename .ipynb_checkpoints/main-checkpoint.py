MAX_LEN = 448
batch_size = 32
d_model = 50
num_heads = 4
N = 2
num_variables = 18 
num_variables += 1 #for no variable embedding while doing padding
d_ff = 100
epochs = 75
learning_rate = 8e-4
drop_out = 0.2
sinusoidal = False
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd
import numpy as np
from utils import MimicDataSetInHospitalMortality, calculate_roc_auc, calculate_auc_prc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

from model import Model
from tqdm import tqdm
from normalizer import Normalizer
from categorizer import Categorizer

train_data_path = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train_listfile.csv"
val_data_path = "/data/datasets/mimic3_18var/root/in-hospital-mortality/val_listfile.csv"

data_dir = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train/"

import pickle

with open('normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

with open('categorizer.pkl', 'rb') as file:
    categorizer = pickle.load(file)
    

mean_variance = normalizer.mean_var_dict
cat_dict = categorizer.category_dict


train_ds = MimicDataSetInHospitalMortality(data_dir, train_data_path, mean_variance, cat_dict, 'training', MAX_LEN)
val_ds = MimicDataSetInHospitalMortality(data_dir, val_data_path, mean_variance, cat_dict, 'validation', MAX_LEN)

train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

model = Model(d_model, num_heads, d_ff, num_variables, N, sinusoidal).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

for epoch in range(epochs):
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        inp = batch['encoder_input']
        mask = batch['encoder_mask']
        y = batch['label']
        outputs = model(inp, mask)
        loss = criterion(outputs, y.float().view(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(f'Epoch {epoch + 1}/{epochs}, Train AUC-ROC: {calculate_roc_auc(model, train_dataloader):.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Validation AUC-ROC: {calculate_roc_auc(model, val_dataloader):.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Validation AUC-PRC: {calculate_auc_prc(model, val_dataloader):.3f}')
    

# Constructing the file path
file_path = f"model_maxlen{MAX_LEN}_batch{batch_size}_dmodel{d_model}_heads{num_heads}_N{N}_vars{num_variables}_dff{d_ff}_epochs{epochs}_lr{learning_rate}_dropout{drop_out}_sinusoidal{sinusoidal}.pth"

# Example usage
torch.save(model.state_dict(), "models/"+ file_path)
