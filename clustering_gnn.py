import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

# 设备配置
# Device configuration (GPU/CPU setup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载和预处理函数
# Data loading and preprocessing functions
def load_data(file_path, label, k, mode, if_norm):
    column_names = [
        'event_index',
        'x_coords', 
        'y_coords',
        'z_coords',
        'time'
    ]
    
    df = pd.read_csv(file_path, header=0, names=column_names)
    
    numeric_cols = ['x_coords', 'y_coords', 'z_coords', 'time']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    df = df.dropna(subset=numeric_cols)
    
    # 数据归一化
    # Data normalization
    # 空间坐标列
    # Spatial coordinates columns
    space_cols = ['x_coords', 'y_coords', 'z_coords']
    # 时间列
    # time_col = 'time'

    if if_norm:
        # 对空间坐标进行标准化
        # Normalize spatial coordinates
        scaler = StandardScaler()
        df[space_cols] = scaler.fit_transform(df[space_cols])

        # 对时间进行单独处理
        # df[time_col] = df[time_col] / 3
    
    data_list = []
    
    for particle_id, group in df.groupby('event_index'):
        features = group[numeric_cols].values.astype(np.float32)
        
        if len(features) < 2:
            continue
            
        x = torch.tensor(features, dtype=torch.float32)
        
        edge_index = manual_knn(features[:, :], k, mode)
        
        data = Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            y=torch.tensor([label], dtype=torch.long).to(device)
        )
        data_list.append(data)
    
    return data_list

def manual_knn(pos, k, mode):
    pos = torch.tensor(pos, dtype=torch.float32)
    num_points = pos.size(0)
    k = min(k, num_points - 1)
    if k < 1:
        return torch.zeros((2, 0), dtype=torch.long)  # 返回标准形状的空边  Return empty edges in standard shape

    def get_edges(feature):
        dist = torch.cdist(feature, feature)
        _, indices = torch.topk(dist, k=k+1, dim=1, largest=False)
        rows = torch.arange(pos.size(0)).view(-1,1).repeat(1, k+1).flatten()
        cols = indices.flatten()
        mask = rows != cols
        edges = torch.stack([rows[mask], cols[mask]], dim=0)
        return edges.unique(dim=1)  # 去重确保边唯一  Remove duplicates to ensure edge uniqueness

    if mode == 1:
        time_feat = pos[:, -1].reshape(-1, 1)
        edges = get_edges(time_feat)
    elif mode == 2:
        edges = get_edges(pos)
    elif mode == 3:
        time_edges = get_edges(pos[:, -1].reshape(-1, 1))
        space_edges = get_edges(pos)
        set_time = {tuple(e) for e in time_edges.t().cpu().numpy()}
        set_space = {tuple(e) for e in space_edges.t().cpu().numpy()}
        intersection = set_time & set_space
        if not intersection:
            return torch.zeros((2, 0), dtype=torch.long)
        edges = torch.tensor(list(intersection)).t()
    else:
        raise ValueError("Invalid mode")

    return edges if edges.numel() > 0 else torch.zeros((2, 0), dtype=torch.long)

# 定义GNN模型
# Define GNN model
class ParticleGNN(torch.nn.Module):
    def __init__(self, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)
        
    def forward(self, x, edge_index, batch):
        # 1. 进行图卷积
        # Perform graph convolution
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        # 2. 全局池化
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # 3. 分类层
        # Classification layer
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

model = ParticleGNN(hidden_channels=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 训练函数
# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# 测试函数
# Testing function
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# 循环
# Loop
k_min = 1
k_max = 6
loop_time = 10
test_acc = np.zeros((loop_time, k_max - k_min + 1))
train_num = 500
date = "20250326_2"
random_state = 40
if_norm = 1
for mode in range(1, 4):            # 1 for time; 2 for position; 3 for both. For knn generation
    index_i = 0
    data_num_str = '2k'
    with tqdm(total=k_max - k_min + 1, desc="k_loop") as pbar_outer:
        for k in range(k_min, k_max + 1):

            # 加载数据集
            # Load the dataset
            file_route = 'D:\\GNN\\simulate_data\\transfer_9435897_files_031472cb'
            proton_data = load_data(file_route + f'\\{data_num_str}proton_Emin1Emax100_digitized_hits_continuous_merged_time_filtered.csv', label=0, k=k, mode=mode, if_norm=if_norm)
            pion_data = load_data(file_route + f'\\{data_num_str}pi-_Emin1Emax100_digitized_hits_continuous_merged_time_filtered.csv', label=1, k=k, mode=mode, if_norm=if_norm)
            dataset = proton_data + pion_data

            index_j = 0
            for j in range(loop_time):
                # 重置模型参数（关键）
                # Reset model parameters (critical)
                model.apply(lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None)

                # 合并并分割数据集
                # Merge and split dataset
                train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=random_state + j)
                val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=random_state + j)

                # 创建数据加载器
                # Create data loader
                train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=64)
                test_loader = DataLoader(test_data, batch_size=64)

                # 训练循环
                # Training loop
                best_val_acc = 0
                patience = 50
                with tqdm(total=train_num, desc=f"k: {k}") as pbar_inner:
                    for epoch in range(train_num):
                        loss = train()
                        train_acc = test(train_loader)
                        val_acc = test(val_loader)
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            torch.save(model.state_dict(), 'best_model.pth')
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                break
                        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                            f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
                        pbar_inner.update(1)
                # 最终测试
                # Final testing
                model.load_state_dict(torch.load('best_model.pth', weights_only=True))
                test_acc[index_j, index_i] = test(test_loader)
                print(f'\nFinal Test Accuracy: {test_acc[index_j, index_i]:.4f}')
                index_j += 1
                # 清理GPU缓存
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pbar_outer.update(1)
            index_i += 1

    save_dir = "D:\\GNN\\results\\" + date
    if mode == 1:
        file_name = f"k_results_kmin_{k_min}_kmax_{k_max}_{data_num_str}_time.txt"
    elif mode == 2:
        file_name = f"k_results_kmin_{k_min}_kmax_{k_max}_{data_num_str}_pos.txt"
    elif mode == 3:
        file_name = f"k_results_kmin_{k_min}_kmax_{k_max}_{data_num_str}_both.txt"

    # 确保目标目录存在
    # Ensure target directory exists
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）  Automatically create directory (if it doesn't exist)

    # 拼接完整的文件路径
    # Construct full file path
    full_path = os.path.join(save_dir, file_name)

    # 保存矩阵为txt文件
    # Save matrix as txt file
    np.savetxt(
        fname=full_path,  # 文件路径  File path
        X=test_acc,         # 要保存的矩阵  Matrix to save
        fmt="%f",
        delimiter=","     # 分隔符（可选，默认是空格）  Delimiter (optional, default is space)
    )
