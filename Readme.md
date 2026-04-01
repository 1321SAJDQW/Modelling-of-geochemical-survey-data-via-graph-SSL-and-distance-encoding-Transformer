# Modelling of Geochemical Survey Data via Graph Self-Supervised Learning and Distance-Encoding Transformer

Geochemical anomaly detection is crucial for mineral resource assessment, but developing effective models to capture complex spatial patterns in geochemical survey data, particularly with limited training samples, remains a significant challenge. This paper proposes a framework that combines graph-based self-supervised learning with Transformer models to detect anomalies related to mineralization. The framework leverages graph convolutional networks (GCNs) to extract local features and employs Transformers to model global spatial dependencies. Additionally, a distance encoding (DE) mechanism is introduced in the Transformer to capture the topology of the geochemical graph.
## Key Modules

The code includes the following key modules:

- **Construction of the geochemical graph**: Edge weights are determined based on the cosine similarity between graph nodes.
- **Distance encoding**: A distance encoding mechanism is designed to help the Transformer capture the topological structure of the geochemical graph.
- **Feature extraction using a combination of GCN and Transformer**: During the pretraining phase of graph self-supervised learning, this allows the model to learn spatial patterns effectively.
- **Fine-tuning phase**: The model is trained using known positive and negative samples to extract mineralization-related geochemical anomalies, improving the model's performance in identifying geochemical anomalies.

## Repository Structure

This repository implements a framework for geochemical anomaly recognition using graph-based self-supervised learning models. The key modules are:

- **Create_Graph.py**: Constructs the geochemical graph.
- **Create_Graph_Weight.py**: Builds the geochemical graph with edge weights.
- **Distance_Encoding_Calculation.py**: Encodes the topological distances between nodes in the graph.
- **Loss.py**: Implements the contrastive loss function for self-supervised learning.
- **Base_GCN_SSL.py**: Implements the base structure for the graph self-supervised learning model.
- **Weighted_GCN_SSL.py**: Enhances the self-supervised learning model by incorporating edge weights into the GCN.
- **Weighted_GCN_Transformer_SSL.py**: Combines weighted GCN with Transformer for improved graph learning.
- **Weighted_GCN_Transformer_DE_SSL.py**: Extends the previous model with a distance encoding mechanism to improve the modeling of graph topology.

## Environment

This code was developed and tested in the following environment:

- **Python**: 3.9
- **PyTorch**: 2.2 (or later versions recommended)
- **CUDA**: 11.8

## Required Dependencies

To ensure the code runs properly, please install the following essential libraries:

- numpy
- pandas
- torch
- torch_geometric
- networkx
- scipy
- sklearn
- matplotlib

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
