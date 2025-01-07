# Graph-based Fraud Detection

This project is focused on detecting fraudulent transactions using a Graph Neural Network (GNN) approach. 
The implementation includes preprocessing transaction data, building a graph from the data, and training a GNN model to classify transactions as fraudulent or not.

## Directory Structure

The project directory is organized as follows:

├───dataset
│       card_feature.csv
│       mechant_feature.csv
│       preprocessed_data.csv
│       data.csv
│       smalldata.csv
│
├───model
│       __init__.py
│       edgeGAT.py
│
├───prepare_data
│       __init__.py
│       build_graph.py
│       data_preprocessing.py
│
train.py
train_small.py
generate.py

### Dataset

- **card_feature.csv**: Preprocessed features for each card, it will be generated itself when running the code.
- **data.csv**: Raw transaction data.
- **merchant_feature.csv**: Preprocessed features for each merchant, it will be generated itself when running the code.
- **preprocessed_data.csv**: Preprocessed transaction data, it will be generated itself when running the code.
- **smalldata.csv**: A smaller subset of transaction data for testing purposes.

### Model

- **edgeGAT.py**: Contains the implementation of the `edgeGAT` model, a Graph Attention Network (GAT) used for classifying transactions based on node and edge features.

### Prepare Data

- **build_graph.py**: Contains functions for constructing graphs from transaction data, including `create_graph_from_sample` and `create_traingraph`.
- **data_preprocessing.py**: Contains functions for preprocessing the raw transaction data, such as `preprocess_data`, `calculate_card_feature`, `calculate_merchant_feature`, and `initialize_data`.

### Root Directory Files

- **train.py**: Script for training and testing the GNN model on the whole data.
- **train.py**: Script for training and testing the GNN model on a smaller data.
- **generate.py**: generate the small data "smalldata.csv" from the big data "data.csv"

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:
- torch
- torch_geometric
- pandas
- scikit-learn
- networkx

## Getting Started

Run the **train.py** if you want to use the whole dataset, else run the **train_small.py** to use a smaller dataset.