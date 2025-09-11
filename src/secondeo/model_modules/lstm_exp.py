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
warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class TimeSeriesDataset:
    """Time Series Dataset Preparation"""
    def __init__(self, sequence_length=10, test_size=0.2, val_size=0.1):
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def create_sequences(self, data, target_col=0):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), :])
            y.append(data[i + self.sequence_length, target_col])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, target_col=0):
        """准备数据"""
        # 确保数据是数值类型
        data = df.values.astype(np.float32)
        
        # 划分训练测试集
        train_size = int(len(data) * (1 - self.test_size))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 标准化特征
        self.scaler_x.fit(train_data)
        train_scaled = self.scaler_x.transform(train_data)
        test_scaled = self.scaler_x.transform(test_data)
        
        # 创建序列
        X_train, y_train = self.create_sequences(train_scaled, target_col)
        X_test, y_test = self.create_sequences(test_scaled, target_col)
        
        # 进一步划分验证集
        val_size = int(len(X_train) * self.val_size)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device).view(-1, 1)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.FloatTensor(y_val).to(device).view(-1, 1)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device).view(-1, 1)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.dropout(out)
        out = self.fc(out)
        return out

class LSTMTrainer:
    """LSTM训练器"""
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
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=100, early_stopping=True):
        """训练模型"""
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
            
            # 早停机制
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
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f'Training completed. Best validation loss: {self.best_val_loss:.6f}')
    
    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def predict(model, X, y_true=None, scaler_x=None, scaler_y=None):
    """进行预测"""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    if y_true is not None and scaler_y is not None:
        # 反标准化预测结果和真实值
        predictions_np = predictions.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # 创建完整的数据矩阵用于反标准化
        pred_full = np.zeros((len(predictions_np), X.shape[2]))
        true_full = np.zeros((len(y_true_np), X.shape[2]))
        
        pred_full[:, 0] = predictions_np.flatten()
        true_full[:, 0] = y_true_np.flatten()
        
        predictions_orig = scaler_y.inverse_transform(pred_full)[:, 0]
        y_true_orig = scaler_y.inverse_transform(true_full)[:, 0]
        
        return predictions_orig, y_true_orig
    
    return predictions.cpu().numpy()

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def plot_predictions(y_true, y_pred, title='Predictions vs Actual'):
    """绘制预测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 1. 生成示例数据（在实际应用中替换为您的数据）
    # 假设有3个特征：目标变量（lagged values）和2个天气特征
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    
    # 生成目标变量（带有趋势和季节性）
    target = 10 + 0.1 * time + 5 * np.sin(2 * np.pi * time / 50) + np.random.normal(0, 1, n_samples)
    
    # 生成天气特征
    temp = 20 + 5 * np.sin(2 * np.pi * time / 100) + np.random.normal(0, 2, n_samples)
    humidity = 60 + 10 * np.sin(2 * np.pi * time / 80) + np.random.normal(0, 3, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'target': target,
        'temperature': temp,
        'humidity': humidity
    })
    
    print("Generated data shape:", df.shape)
    print(df.head())
    
    # 2. 准备数据
    seq_length = 20
    dataset = TimeSeriesDataset(sequence_length=seq_length, test_size=0.2, val_size=0.1)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.prepare_data(df, target_col=0)
    
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    print(f"Test data: {X_test.shape}, {y_test.shape}")
    
    # 3. 创建模型
    input_size = X_train.shape[2]  # 特征数量
    hidden_size = 50
    num_layers = 2
    output_size = 1
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 训练模型
    trainer = LSTMTrainer(model, learning_rate=0.001, patience=20)
    trainer.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=100)
    
    # 绘制训练损失
    trainer.plot_losses()
    
    # 5. 在测试集上进行预测
    test_predictions, test_actual = predict(
        model, X_test, y_test, dataset.scaler_x, dataset.scaler_x
    )
    
    # 6. 计算评估指标
    print("\nTest set performance:")
    metrics = calculate_metrics(test_actual, test_predictions)
    
    # 7. 绘制预测结果
    plot_predictions(test_actual, test_predictions, 'Test Set Predictions vs Actual')
    
    # 8. 进行未来预测（多步预测）
    def predict_future(model, last_sequence, steps=30, scaler=None):
        """预测未来多个时间步"""
        model.eval()
        predictions = []
        current_sequence = last_sequence.clone()
        
        with torch.no_grad():
            for _ in range(steps):
                # 预测下一个时间步
                pred = model(current_sequence.unsqueeze(0))
                predictions.append(pred.item())
                
                # 更新序列：移除第一个时间步，添加预测结果
                # 注意：这里假设第一个特征是目标变量
                new_features = current_sequence[1:, :].clone()
                new_last_step = torch.cat([
                    pred.view(1, 1),  # 预测的目标值
                    current_sequence[-1, 1:].unsqueeze(0)  # 其他特征保持不变（实际中可能需要更新）
                ], dim=1)
                
                current_sequence = torch.cat([new_features, new_last_step], dim=0)
        
        # 反标准化预测结果
        if scaler is not None:
            pred_array = np.array(predictions).reshape(-1, 1)
            # 创建完整矩阵用于反标准化
            full_pred = np.zeros((len(pred_array), current_sequence.shape[1]))
            full_pred[:, 0] = pred_array.flatten()
            predictions = scaler.inverse_transform(full_pred)[:, 0]
        
        return np.array(predictions)
    
    # 使用最后一段序列进行未来预测
    last_sequence = X_test[-1]  # 取测试集最后一个序列
    future_steps = 30
    future_predictions = predict_future(model, last_sequence, future_steps, dataset.scaler_x)
    
    print(f"\nFuture {future_steps} steps predictions:")
    print(future_predictions)
    
    # 绘制未来预测
    plt.figure(figsize=(12, 6))
    plt.plot(test_actual[-50:], label='Last 50 Actual', marker='o')
    plt.plot(range(len(test_actual[-50:]), len(test_actual[-50:]) + future_steps), 
             np.concatenate([[test_actual[-1]], future_predictions]), 
             label='Future Predictions', marker='o', linestyle='--')
    plt.title('Future Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()