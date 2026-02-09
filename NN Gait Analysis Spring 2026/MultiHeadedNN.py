"Multi-Headed Neural Network"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


class MultiHeadNN(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: list[int],
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True):
        
        super(MultiHeadNN, self).__init__()
        
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
        shared_layers = [nn.Linear(hidden_size[0], hidden_size[1])]
        if use_batch_norm:
            shared_layers.append(nn.BatchNorm1d(hidden_size[1]))
        shared_layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size[1], 64)
        ])
        if use_batch_norm:
            shared_layers.append(nn.BatchNorm1d(64))
        shared_layers.append(nn.ReLU())
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Output head 1: Binary classifier
        self.binary_output = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Output head 2: Percentage (regression)
        self.percentage_output = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
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
        x = self.shared(x)
        binary_out = self.binary_output(x)
        percentage_out = self.percentage_output(x)
        return binary_out, percentage_out
    
    
def prepare_data(
    X, 
    y_bi,
    y_per,
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
    if hasattr(y_bi, 'values'):
        y_bi = y_bi.values
    if hasattr(y_per, 'values'):
        y_per  = y_per.values
    
    # Convert to tensors (use float32 for consistency)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_bi_tensor = torch.tensor(y_bi, dtype=torch.float32)
    y_per_tensor = torch.tensor(y_per, dtype=torch.float32)
    
    # Reshape y if needed for binary classification
    if len(y_bi_tensor.shape) == 1:
        y_bi_tensor = y_bi_tensor.unsqueeze(1)
    if len(y_per_tensor.shape) == 1:
        y_per_tensor = y_bi_tensor.unsqueeze(1)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_bi_tensor, y_per_tensor)
    
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
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    patience: int = 10,
    save_path: str = 'best_MH_model.pth',
    loss_weights: tuple = (1.0, 1.0)
) -> dict:
    """
    Train the model with validation and early stopping for multi-input, multi-output
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader (yields (input1, input2), (binary_target, percentage_target))
        val_loader: Validation data loader
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (GPU/CPU)
        patience: Number of epochs to wait before early stopping
        save_path: Path to save the best model
        loss_weights: Tuple of (binary_weight, percentage_weight) for loss combination
    
    Returns:
        Dictionary containing training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    print(f"Training on: {device}")
    
    # Loss functions for each output
    binary_criterion = nn.L1Loss()  # Binary classification
    percentage_criterion = nn.MSELoss()  # Regression for percentage
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_binary_loss': [],
        'train_percentage_loss': [],
        'val_loss': [],
        'val_binary_loss': [],
        'val_percentage_loss': [],
        'train_binary_acc': [],
        'val_binary_acc': [],
        'train_percentage_mae': [],  # Mean Absolute Error for percentage
        'val_percentage_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_binary_loss = 0.0
        train_percentage_loss = 0.0
        train_binary_correct = 0
        train_binary_total = 0
        train_percentage_mae = 0.0
        
        for X_batch, y_bi_batch, y_per_batch in train_loader:
            # Move data to device
            input_batch = X_batch.to(device)
            binary_target = y_bi_batch.to(device)
            percentage_target = y_per_batch.to(device)
            
            # Normalize percentage target to [0, 1] if it's in [0, 100]
            if percentage_target.max() > 1.0:
                percentage_target = percentage_target / 100.0
            
            # Forward pass
            binary_output, percentage_output = model(input_batch)
            
            # Calculate losses
            loss_binary = binary_criterion(binary_output, binary_target)
            loss_percentage = percentage_criterion(percentage_output, percentage_target)
            
            # Combined loss with weights
            loss = loss_weights[0] * loss_binary + loss_weights[1] * loss_percentage
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_size = input_batch.size(0)
            train_loss += loss.item() * batch_size
            train_binary_loss += loss_binary.item() * batch_size
            train_percentage_loss += loss_percentage.item() * batch_size
            
            # Binary accuracy
            binary_predictions = (binary_output > 0.5).float()
            train_binary_correct += (binary_predictions == binary_target).sum().item()
            train_binary_total += binary_target.numel()
            
            # Percentage MAE (convert back to percentage scale)
            train_percentage_mae += (torch.abs(percentage_output - percentage_target) * 100).sum().item()
        
        # Calculate average training metrics
        dataset_size = len(train_loader.dataset)
        avg_train_loss = train_loss / dataset_size
        avg_train_binary_loss = train_binary_loss / dataset_size
        avg_train_percentage_loss = train_percentage_loss / dataset_size
        train_binary_accuracy = train_binary_correct / train_binary_total
        avg_train_percentage_mae = train_percentage_mae / train_binary_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_binary_loss = 0.0
        val_percentage_loss = 0.0
        val_binary_correct = 0
        val_binary_total = 0
        val_percentage_mae = 0.0
        
        with torch.no_grad():
            for X_batch, y_bi_batch, y_per_batch in val_loader:
                # Move data to device
                input_batch = X_batch.to(device)
                binary_target = y_bi_batch.to(device)
                percentage_target = y_per_batch.to(device)
                
                # Normalize percentage target
                if percentage_target.max() > 1.0:
                    percentage_target = percentage_target / 100.0
                
                # Forward pass
                binary_output, percentage_output = model(input_batch)
                
                # Calculate losses
                loss_binary = binary_criterion(binary_output, binary_target)
                loss_percentage = percentage_criterion(percentage_output, percentage_target)
                loss = loss_weights[0] * loss_binary + loss_weights[1] * loss_percentage
                
                # Track metrics
                batch_size = input_batch.size(0)
                val_loss += loss.item() * batch_size
                val_binary_loss += loss_binary.item() * batch_size
                val_percentage_loss += loss_percentage.item() * batch_size
                
                # Binary accuracy
                binary_predictions = (binary_output > 0.5).float()
                val_binary_correct += (binary_predictions == binary_target).sum().item()
                val_binary_total += binary_target.numel()
                
                # Percentage MAE
                val_percentage_mae += (torch.abs(percentage_output - percentage_target) * 100).sum().item()
        
        # Calculate average validation metrics
        val_dataset_size = len(val_loader.dataset)
        avg_val_loss = val_loss / val_dataset_size
        avg_val_binary_loss = val_binary_loss / val_dataset_size
        avg_val_percentage_loss = val_percentage_loss / val_dataset_size
        val_binary_accuracy = val_binary_correct / val_binary_total
        avg_val_percentage_mae = val_percentage_mae / val_binary_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_binary_loss'].append(avg_train_binary_loss)
        history['train_percentage_loss'].append(avg_train_percentage_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_binary_loss'].append(avg_val_binary_loss)
        history['val_percentage_loss'].append(avg_val_percentage_loss)
        history['train_binary_acc'].append(train_binary_accuracy)
        history['val_binary_acc'].append(val_binary_accuracy)
        history['train_percentage_mae'].append(avg_train_percentage_mae)
        history['val_percentage_mae'].append(avg_val_percentage_mae)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Binary Acc: {train_binary_accuracy:.4f}, MAE Loss : {avg_train_percentage_loss:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Binary Acc: {val_binary_accuracy:.4f}, MAE Loss : {avg_val_percentage_loss:.2f}%")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_binary_acc': val_binary_accuracy,
                'val_percentage_mae': avg_val_percentage_mae
            }, save_path)
            print("  → Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Best binary accuracy: {checkpoint['val_binary_acc']:.4f}")
    print(f"Best percentage MAE: {checkpoint['val_percentage_mae']:.2f}%")
    
    return history


def predict(
    model: nn.Module, 
    X, 
    device: Optional[torch.device] = None,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on new data for both binary classification and percentage regression
    
    Args:
        model: Trained PyTorch model
        X: Input features
        device: Device to run inference on
        threshold: Classification threshold for binary output
    
    Returns:
        Tuple of (binary_predictions, percentage_predictions) as numpy arrays
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
        # Model returns two outputs: (binary_output, percentage_output)
        binary_output, percentage_output = model(X_tensor)
        
        # Binary predictions (apply threshold)
        binary_predictions = (binary_output > threshold).float()
        
        # Percentage predictions (already 0-1 from sigmoid, multiply by 100 for percentage)
        percentage_predictions = percentage_output * 100
    
    return binary_predictions.cpu().numpy(), percentage_predictions.cpu().numpy()