"Using Pytorch and testing the differences between that and TensorFlow"

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as opt
"Going to watch the Youtube tutorial and see if this works better for my needs"

class TorchNN(torch.nn.Module):
    
    def __init__(self, input_size = float, nodes_num = float, output_size = float):
        super(TorchNN, self).__init__()
        
        self.i_s = input_size
        self.nod = nodes_num
        self.o_s = output_size
        
        self.lay1 = nn.Linear(self.i_s, self.nod)
        self.lay2 = nn.Linear(self.nod, self.nod)
        self.lay3 = nn.Linear(self.nod, self.o_s)
         
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.ReLU(self.lay1(x))
        x = nn.ReLU(self.lay2(x))
        x = self.lay3(x)
        
        return x
    
def data_conversion(X, Y, bs):
    X_tensor = torch.tensor(X.values, dtype=torch.float64)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float64)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size= bs, shuffle=True)
    return X_tensor, Y_tensor, dataloader

def training(x, y, dataloader, model, num_epochs, lr):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = opt.Adam(model.parameters(), lr = lr)
    num_epochs = 10
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
    
        for x, y in dataloader:
            y = y.float()
    

            logits = model(x)
    
            # 2. Compute loss
            loss = criterion(logits, y)
    
            # 3. Backpropagation
            optimizer.zero_grad()
            loss.backward()
    
            # 4. Update weights
            optimizer.step()
    
            epoch_loss += loss.item()
    
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    return model


