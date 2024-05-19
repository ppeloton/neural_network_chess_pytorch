import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
import datetime


from common import common

model = common.SupNetwork()

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn_value = nn.MSELoss()
loss_fn_policy = nn.CrossEntropyLoss()

epochs = 512
batch_size = 16
device = "cpu"

inputData = np.load("positions.npy")
policyOutcomes = np.load("moveprobs.npy")
valueOutcomes = np.load("outcomes.npy")

dataset = common.CustomDataset(inputData, policyOutcomes, valueOutcomes)
training_dataloader = DataLoader(dataset=dataset, 
                              batch_size=batch_size, 
                              num_workers=1, 
                              shuffle=True) 

common.trainModel(model, training_dataloader, epochs, optimizer, loss_fn_policy,
                          loss_fn_value, "cpu")
        
torch.save(model.state_dict(), "model_sup.pth")