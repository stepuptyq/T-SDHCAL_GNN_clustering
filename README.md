# Particle Identification using Graph Neural Networks

This repository contains code for processing particle collision data and training a Graph Neural Network (GNN) model for particle identification tasks.

## Main Components

### 1. Data Processing (`process_particle_data.py`)
Key features:
- Filters particle events using IQR-based outlier detection
- Performs data standardization:
  - Spatial coordinates normalized by 1000
  - Time values normalized by 5
- Handles grouped particle events using event indices

### 2. GNN Training (`gnn_training.py`)
Key components:
- Graph construction using k-NN with 3 modes:
  1. Time-based adjacency
  2. Spatial-based adjacency 
  3. Intersection of time and space adjacencies
- GNN architecture:
  - 3-layer GCN with ReLU activation
  - Global mean pooling
  - Classification head
- Training features:
  - Early stopping with patience=50
  - Multiple k-NN experiments (k=1-6)
  - 10-fold cross-validation
  - GPU acceleration support

## Requirements
- Python 3.7+
- Dependencies:
  ```bash
  pandas==1.3.5
  numpy==1.21.6
  torch==1.13.1
  torch-geometric==2.2.0
  scikit-learn==1.0.2
  tqdm==4.64.1
  ```

## Usage

### 1. Data Processing
```python
from process_particle_data import process_particle_data

process_particle_data(
    input_file="path/to/raw_data.csv",
    output_file="path/to/processed_data.csv"
)
```

### 2. Model Training
Configure parameters in the training script:
```python
# Experiment parameters
k_min = 1      # Minimum k-NN neighbors
k_max = 6      # Maximum k-NN neighbors
loop_time = 10 # Number of repetitions
train_num = 500# Maximum epochs
if_norm = 1    # Enable normalization (1=True)
mode = 1       # 1=Time, 2=Space, 3=Both
```

Run training:
```python
python gnn_training.py
```

### 3. Input Data Format
Required CSV columns:
- `event_index`: Particle event identifier
- `x_coords`, `y_coords`, `z_coords`: Spatial coordinates
- `time`: Timestamp

## Results
Output files contain test accuracy matrices with shape:
- Rows: Number of repetitions (`loop_time`)
- Columns: k-NN parameters (k_max - k_min + 1)

Example output path:
```
results/20250326_2/k_results_kmin_1_kmax_6_2k_time.txt
```

## Configuration Notes
1. Update file paths in both scripts to match your system
2. Adjust IQR multiplier (1.5) in data processing as needed
3. Modify hidden_channels (default=128) in ParticleGNN for model capacity
4. Tune learning rate (default=0.001) and batch size (default=64)

## Key Hyperparameters
| Parameter       | Description                     | Default Value |
|-----------------|---------------------------------|---------------|
| hidden_channels | GCN layer dimensionality        | 128           |
| lr              | Learning rate                   | 0.001         |
| k_min/k_max     | k-NN neighbors range            | 1-6           |
| patience        | Early stopping threshold        | 50            |
| batch_size      | Training batch size             | 64            |
