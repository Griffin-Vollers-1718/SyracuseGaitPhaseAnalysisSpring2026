"All Planes Neural Network using MultiHeaded Architecture"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


class BinaryClassifier(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: list[int],
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True):
        
        super(BinaryClassifier, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Input branch with batch normalization
        layers = [nn.Linear(input_size, hidden_size[0])]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size[0]))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.input_branch = nn.Sequential(*layers)
        
        # Shared layers with batch normalization
        hidden_layers = [nn.Linear(hidden_size[0], hidden_size[1])]
        if use_batch_norm:
            hidden_layers.append(nn.BatchNorm1d(hidden_size[1]))
        hidden_layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size[1], 64)
        ])
        if use_batch_norm:
            hidden_layers.append(nn.BatchNorm1d(64))
        hidden_layers.append(nn.ReLU())
        
        self.hidden = nn.Sequential(*hidden_layers)
        
        #Output
        self.output = nn.Sequential(
            nn.Linear(64, 1),
            )
        # outputs the pure logits
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
    
    def forward(self, input1):
        x = self.input_branch(input1)
        x = self.hidden(x)
        out = self.output(x)
        return out
    
def prepare_data(
    X, 
    y, 
    batch_size: int,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Convert data to PyTorch tensors and create train/validation dataloaders
    
    Args:
        X: Input features (pandas DataFrame or numpy array)
        y: Target labels (pandas Series or numpy array)
        batch_size: Batch size for training
        train_split: Proportion of data to use for training
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    # Convert to numpy if pandas
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Convert to tensors (use float32 for consistency)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Reshape y if needed for binary classification
    if len(y_tensor.shape) == 1:
        y_tensor = y_tensor.unsqueeze(1)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train/validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, val_dataset


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: Optional[torch.device] = None,
    patience: int = 10,
    save_path: str = 'best_model.pth',
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
        optimizer, mode='min', factor=0.5, patience=5
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
        avg_train_loss = train_loss / len(train_loader.dataset)
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
        avg_val_loss = val_loss / len(val_loader.dataset)
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
            torch.save(model, 'BinaryClassifier.pth')
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