# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim, cuda
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim, cuda
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import csv
import os
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score

from tqdm import tqdm

from data.datasets import DeClareDataset
from model.deClare import DeClareModel
from metrics.evaluation import DeClareEvaluation


# %%
torch.cuda.is_available()

# %% [markdown]
# # Load Datasets

# %%
SNOPES_LOC = "./Datasets/Snopes/snopes.tsv"
#consists of rumors analyzed on the Snopes website along with their credibility labels (true or false), 
#sets of reporting articles, and their respective web sources

POLITIFACT_LOC = "./Datasets/PolitiFact/politifact.tsv"

glove_data_file = "./Glove/glove.6B.100d.txt"


# %%
snopes = DeClareDataset(SNOPES_LOC, glove_data_file)


# %%
snopes.news_df.head()

# %% [markdown]
# ## Preliminary Analysis

# %%
batch_size = 64
nb_lstm_units = 64
random_seed = 42

val_split = 0.1
test_split = 0.1
shuffle_dataset = False


# %%
# Creating data indices for training and validation splits:
dataset_size = len(snopes)
indices = list(range(dataset_size))

val_split = int(np.floor(val_split * dataset_size))
test_split = val_split + int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[test_split:], indices[:val_split], indices[val_split:test_split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)


# %%
train_dataloader = DataLoader(snopes, batch_size, sampler=train_sampler)
val_dataloader = DataLoader(snopes, len(val_indices), sampler=val_sampler)
test_dataloader = DataLoader(snopes, len(test_indices), sampler=test_sampler)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
declare = DeClareModel(snopes.initial_embeddings, snopes.claim_source_vocab_size, 
                       snopes.article_source_vocab_size, nb_lstm_units, device)

declare.to(device)


# %%
optimizer = optim.Adam(declare.parameters(), lr=0.005)
BCEloss = nn.BCELoss()


# %%
losses = {}
num_epochs = 20
reg_lambda = 0.0002
writer = SummaryWriter()



for epoch in range(num_epochs):
    losses[epoch] = []
    for data_sample in tqdm(train_dataloader):
        
        declare.zero_grad()
        idx = np.argsort(-data_sample[3])

        for i in range(len(data_sample)):
            data_sample[i] = data_sample[i][idx].to(device)

        out = declare(data_sample[0], data_sample[1], data_sample[2], data_sample[3], data_sample[4], data_sample[5])
        unsqueezed_data = data_sample[6].float().unsqueeze(-1)
        loss = BCEloss(out, unsqueezed_data)

        l2_reg = None
        for param in declare.named_parameters():
            if 'dense' in param[0] and 'weight' in param[0]:
                if l2_reg is None:
                    l2_reg = param[1].norm(2)
                else:
                    l2_reg = l2_reg + param[1].norm(2)

        total_loss = loss + reg_lambda*l2_reg

        total_loss.backward()
        optimizer.step()

        writer.add_scalar('total loss', total_loss, epoch)
        writer.add_scalar('regularization loss', l2_reg, epoch)
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('pad embedding', declare.word_embeddings(torch.tensor([0]).to(device)).data.mean())
        
        losses[epoch].append(total_loss.data)


# %%
plt.plot(losses[0])


# %%
torch.save(declare.state_dict(), 'demo_model')
declare.load_state_dict(torch.load('demo_model'))


# %%
snopes_eval = DeClareEvaluation(declare, test_dataloader, device)
labels, preds = snopes_eval.claim_wise_accuracies()


# %%
true_claim_indices = np.where(labels==1)
false_claim_indices = np.where(labels==0)


# %%
accuracy_score(labels[true_claim_indices], preds[true_claim_indices]>0.5)


# %%
accuracy_score(labels[false_claim_indices], preds[false_claim_indices]>0.5)


# %%
roc_auc_score(labels, preds)


# %%
potifact_df = pd.read_csv(POLITIFACT_LOC, sep='\t', header=None)


# %%
potifact_df.head()


# %%
snopes_df = pd.read_csv(SNOPES_LOC, sep='\t', header=None)
snopes_df.head()



# %%

# %%
