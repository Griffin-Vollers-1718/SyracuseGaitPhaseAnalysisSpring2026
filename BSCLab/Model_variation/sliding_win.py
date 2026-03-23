
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window, pred_len, scaler=None):
        self.window = window
        self.pred_len = pred_len

        if scaler is None:
            self.X_scaler = StandardScaler().fit(X)
            self.y_scaler = StandardScaler().fit(y)
        else:
            self.X_scaler, self.y_scaler = scaler

        # Store as tensors directly — no more self.len needed
        self.X = torch.FloatTensor(self.X_scaler.transform(X))
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X) - self.window - self.pred_len + 1

    def __getitem__(self, idx):
        X_window = self.X[idx : idx + self.window]
        y_window = self.y[idx + self.window : idx + self.window + self.pred_len]
        return X_window, y_window

    def get_scaler(self):
        return self.X_scaler, self.y_scaler
