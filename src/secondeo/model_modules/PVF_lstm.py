import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
from typing import Union
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def create_torchdata(
        X_train: pd.DataFrame, 
        y_train: Union[pd.Series, pd.DataFrame],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, pd.DataFrame],
        val_size: float,
        device: torch.device,
        normalize: bool = True
) -> dict[str, DataLoader]:
    """Create DataLoader from training and testing data.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series or pd.DataFrame): Testing target.
        val_size (float): Proportion of training data to use for validation.
        device (torch.device): Device to load tensors onto.
        normalize (bool): Whether to apply Min-Max normalization.
    """
    # Convert to numpy arrays
    X_train, y_train, X_test, y_test = map(np.array, (X_train, y_train, X_test, y_test))
    
    # Apply normalization if specified ------------------------- TODO: if need to normalize y?
    if normalize:
        scaler_X = MinMaxScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        # Note: y normalization can be added if needed
    
    # Reshape X to 3D (samples, timesteps, features) if needed
    if X_train.ndim == 2:
        X_train = X_train[:, np.newaxis, :]  # (samples, 1, features)
        X_test = X_test[:, np.newaxis, :]  # (samples, 1, features)
    elif X_train.ndim != 3:
        raise ValueError(f"X must be 2D or 3D, got shape {X_train.shape}")
        
    # Generate validation set from training data
    val_size = int(len(X_train) * val_size)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size:]
          
    # Convert to PyTorch tensors and move to device
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device).view(-1, 1)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device).view(-1, 1)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device).view(-1, 1)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


class LSTMModel(nn.Module):
    """LSTM Model for PV Prediction"""
    def __init__(self, params: dict):
        super(LSTMModel, self).__init__()
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.input_size = params["input_size"]
        self.output_size = params["output_size"]
        self.dropout = params["dropout"]

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True, dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM forward propagation
        out, _ = self.lstm(x, (h0, c0))
        # Only take the output from the last time step
        out = out[:, -1, :]
        # Fully connected layer
        out = self.dropout(out)
        out = self.fc(out)

        return out


class LSTMTrainer:
    """LSTM Model Trainer with Early Stopping and LR Scheduler"""
    def __init__(self, model, learning_rate=0.001, patience=10):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience//2, factor=0.5)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience = patience
        self.counter = 0
    
    def train_epoch(self, train_loader):
        """Training for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_train_batch, y_train_batch in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X_train_batch)
            loss = self.criterion(outputs, y_train_batch)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """ Validate the model """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                outputs = self.model(X_val_batch)
                loss = self.criterion(outputs, y_val_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=100, early_stopping=True):
        """Training the model"""
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                self.counter = 0
            else:
                self.counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if early_stopping and self.counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Load the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f'Training completed. Best validation loss: {self.best_val_loss:.6f}')
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def predict(self, X_test):
        """Make predictions with the trained model"""  # ---------------------- TODO: USE BATCH
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()
        return predictions.flatten()
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="LSTM model for PV forecasting.")
    argparser.add_argument(
        "-execute_optimize",
        type=bool,
        default=True,
        help="Whether to run hyperparameter optimization with Optuna."
    )
    args = argparser.parse_args()
    execute_optimize = args.execute_optimize

    # Load params
    with open("src_pandas/params_modules/data_param.yaml", 'r') as file:
        data_params = yaml.safe_load(file)
    with open("src_pandas/params_modules/model_param.yaml", 'r') as file:
        model_params = yaml.safe_load(file)
    datetime_column = data_params['datetime_column']
    target_column = data_params['target_column']
    feature_table_path = data_params['feature_table_save_path']
    train_start_date = data_params['train_start_date']
    train_end_date = data_params['train_end_date']
    test_start_date = data_params['test_start_date']
    test_end_date = data_params['test_end_date']
    lstm_params = model_params['LSTM']

    # Load data
    df = pd.read_csv(feature_table_path, index_col=0)  # Pay attention to exclude time column

    train_df = df[(df.index >= train_start_date) & (df.index <= train_end_date)]
    test_df = df[(df.index >= test_start_date) & (df.index <= test_end_date)]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Create DataLoaders
    data_dict = create_torchdata(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, val_size=lstm_params["val_size"],
        device=device, normalize=False
    )

    X_train, y_train, X_val, y_val = data_dict["X_train"], data_dict["y_train"], data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    print(f"Test data: {X_test.shape}, {y_test.shape}")  # ---------------------- DIFFERENCE: THE MIDDLE COL IS ONE, THE EXAMPLE IS 20

    # Train the model
    model_params = {
        "input_size": X_train.shape[2],
        "output_size": y_train.shape[1],
        "hidden_size": lstm_params["hidden_dim"],
        "num_layers": lstm_params["num_layers"],
        "dropout": lstm_params["dropout"]
    }
    model = LSTMModel(params=model_params).to(device)

    trainer = LSTMTrainer(model, learning_rate=lstm_params["learning_rate"], patience=lstm_params["patience"])
    trainer.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                  batch_size=lstm_params["batch_size"],
                  epochs=lstm_params["epochs"],
                  early_stopping=True)
    trainer.plot_losses()

    # Evaluate on test set
    predictions = trainer.predict(X_test)
    test_rmse = np.sqrt(np.mean((predictions - y_test.cpu().numpy().flatten())**2))
    print(f'Test RMSE: {test_rmse:.6f}')

    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.cpu().numpy(), label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('LSTM Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('PV Output')
    plt.legend()
    plt.grid(True)
    plt.show()
