# Particle Identification with Graph Neural Networks

A Graph Neural Network (GNN) implementation for particle identification using detector hit data. Distinguishes between π⁻ (pion) and proton particles through geometric deep learning on 3D spatial-temporal detector events.

## Key Features

🔭 **Event-based Processing**  
- Processes complete detector events as graph structures
- Spatial-temporal feature encoding (x,y,z coordinates + hit timing)
- Dynamic KNN graph construction with time window constraints

🛡️ **Leakage-proof Design**  
- Strict event-level stratified splitting
- Separate preprocessing pipelines for train/validation/test sets
- Feature normalization fitted exclusively on training data

🧠 **Hybrid Graph Architecture**  
- Combines GATConv, TAGConv, and GraphConv layers
- Graph normalization and dual pooling (max + mean)
- Robustness to variable-sized inputs through graph pooling

## Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/particle-gnn.git
cd particle-gnn
```

2. **Create conda environment**  
```bash
conda create -n particlegnn python=3.8
conda activate particlegnn
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place raw CSV files in `/data/raw` directory:
   - `20kpi-_Emin1Emax100_digitized_hits_continuous_merged.csv`
   - `20kproton_Emin1Emax100_digitized_hits_continuous_merged.csv`

2. Preprocess data:
```python
from data_processing import load_and_split_data
raw_datasets = load_and_split_data(pion_path, proton_path)
```

### Training
Configure hyperparameters in `config.yaml`:
```yaml
hidden_dim: 256
batch_size: 64
lr: 3e-4
epochs: 100
knn_k: 8
```

Start training:
```bash
python train.py
```

### Evaluation
Generate performance reports:
```bash
python evaluate.py --model_checkpoint best_model.pth
```
