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
