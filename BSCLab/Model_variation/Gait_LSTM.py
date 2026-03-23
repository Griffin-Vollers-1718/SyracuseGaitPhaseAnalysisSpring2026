# This is where the LSTM Model will be made and created
# Will use the same architecure as the Binary Classifier


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from sliding_win import TimeSeriesDataset


class GaitLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 ):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first=True,
            dropout = 0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

        self._initialize_weights()
    
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        self.hidden = (h_n.detach(), c_n.detach())
        out = self.fc(out[:, -1, :])  # (batch, output_size)
        return out.unsqueeze(-1)

    def reset_hidden(self):
        self.hidden = None
    
def prepare_data(
    X, 
    y, 
    batch_size: int,
    window: int = 50,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    dataset = TimeSeriesDataset(X, y, window=window, pred_len=1)

    # Sequential split — preserves time order
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle time series data
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    return train_loader, val_loader

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: Optional[torch.device] = None,
    patience: int = 10,
    save_path: str = './LSTM_model.pth',
) -> dict:
    """
    Train the model with validation and early stopping
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (GPU/CPU)
        patience: Number of epochs to wait before early stopping
        save_path: Path to save the best model
    
    Returns:
        Dictionary containing training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    print(f"Training on: {device}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * x_batch.size(0)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.numel()
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader.dataset) # type: ignore
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * x_batch.size(0)
                predictions = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.numel()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset) # type: ignore
        val_accuracy = val_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,

            }, save_path)
            print(" Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    
    return history

def calc_metrics(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).cpu().numpy()
            all_preds.append(outputs)
            all_targets.append(batch_y.numpy())
     
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
     
    # Calculate metrics
    rmse = np.sqrt(np.mean((preds - targets)**2))
    mae = np.mean(np.abs(preds - targets))
    mape = np.mean(np.abs((preds - targets) / (targets + 1e-2)))
    # add a small constant to prevent explosion when targets are near zero
     
    return  rmse.item(), mae.item(), mape.item(), preds, targets


def predict(
    model: nn.Module, 
    X, 
    device: Optional[torch.device] = None,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Make predictions on new data
    
    Args:
        model: Trained PyTorch model
        X: Input features
        device: Device to run inference on
        threshold: Classification threshold
    
    Returns:
        Predictions as numpy array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Convert to tensor
    if hasattr(X, 'values'):
        X = X.values
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
    
    return predictions.cpu().numpy()