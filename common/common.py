import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
import datetime

class SupNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(21, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 128)

        self.policyLayer = nn.Linear(128, 28)        
        self.valueLayer = nn.Linear(128, 1)
        
    def forward(self, x):
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = torch.relu(self.layer3(out))
        out = torch.relu(self.layer4(out))
        out = torch.relu(self.layer5(out))
    
        policyOut = self.policyLayer(out)        
        valueOut = torch.tanh(self.valueLayer(out))

        return policyOut, valueOut

class CustomDataset(Dataset):
    def __init__(self, inputData, policyOutcomes, valueOutcomes):
            
        self.inputData = torch.from_numpy(inputData).type(torch.float)
        
        tmp = [np.where(row == np.max(row))[0][0] for row in policyOutcomes]
        self.policyOutcomes = torch.tensor(tmp)
        
        self.valueOutcomes = torch.from_numpy(valueOutcomes).type(torch.float)

    def __getitem__(self, index):
        inputData = self.inputData[index]
        policyOutcomes = self.policyOutcomes[index]
        valueOutcomes = self.valueOutcomes[index]

        return inputData, policyOutcomes, valueOutcomes

    def __len__(self):
        return len(self.inputData)

def trainModel(model, dataloader, epochs, optimizer, loss_fn_policy, loss_fn_value,
               device):
    
    for epoch in range(1, epochs + 1):
        loss_train = 0.0
        value_loss_train = 0.0
        policy_loss_train = 0.0
        for x, policyTarget, valueTarget in dataloader:
        
            x = x.to(device=device)
            policyTarget = policyTarget.to(device=device)
            
            valueTarget = valueTarget.view((len(valueTarget), 1))
            valueTarget = valueTarget.to(device=device)

            policyOut, valueOut = model(x)

            policyLoss = loss_fn_policy(policyOut, policyTarget)
            valueLoss = loss_fn_value(valueOut,valueTarget) 
            loss = policyLoss + valueLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            value_loss_train += valueLoss
            policy_loss_train += policyLoss

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}, Value loss {}, Policy loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(dataloader),
                value_loss_train / len(dataloader),
                policy_loss_train / len(dataloader)))