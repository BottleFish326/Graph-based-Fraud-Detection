import torch
import pandas as pd
import networkx as nx
from sklearn.utils import resample
from torch_geometric.utils import from_networkx

def downsample_data(data):
    """
    Downsample the data to balance the proportion of positive (fraud) and negative (non-fraud) samples.

    Parameters:
    data (pd.DataFrame): The original transaction data.

    Returns:
    pd.DataFrame: The downsampled data with balanced positive and negative samples.
    """
    fraud_data = data[data['Is Fraud?'] == 1]
    non_fraud_data = data[data['Is Fraud?'] == 0]
    non_fraud_data = resample(non_fraud_data, 
                                replace=False, 
                                n_samples=len(fraud_data)*2)
    sampled_data = pd.concat([fraud_data, non_fraud_data])
    return sampled_data

def create_graph_from_sample(data, card_feature, merchant_feature):
    """
    Create a graph from the sampled transaction data.

    Parameters:
    data (pd.DataFrame): The sampled transaction data.
    card_feature (dict): Dictionary containing features for each card.
    merchant_feature (dict): Dictionary containing features for each merchant.

    Returns:
    nx.MultiDiGraph: The created graph with nodes and edges representing cards and merchants.
    bool: Indicator of whether the graph creation was successful or not.
    """
    # print("merchant_feature")
    # print(merchant_feature)
    G = nx.MultiDiGraph()
    unique_cards = data['Card_id'].unique()
    unique_merchants = data['Merchant Name'].unique()
    # print("unique_merchants")
    # print(unique_merchants)
    if len(unique_cards) + len(unique_merchants) >= len(data) * 1.8:
        return G, False
    for card in unique_cards:
        G.add_node(card, type='card', features=card_feature[card])
    for merchant in unique_merchants:
        # print("merchant")
        # print(merchant)
        # print("merchant_feature")
        # print(merchant_feature[merchant])
        G.add_node(merchant, type='card', features=merchant_feature[merchant])
    for index, row in data.iterrows():
        edge_features = {
            'Year': row['Year'],
            'Month': row['Month'],
            'Day': row['Day'],
            'Time': row['Time'],
            'Amount': row['Amount'],
            'Use Chip': row['Use Chip'],
            'Merchant City': row['Merchant City'],
            'Merchant State': row['Merchant State'],
            'Zip': row['Zip'],
            'MCC': row['MCC'],
            'Errors?': row['Errors?'],
            'Is Fraud?': row['Is Fraud?']
        }
        G.add_edge(row['Card_id'], row['Merchant Name'], features=edge_features, label=row['Is Fraud?'])
    return G, True

def create_traingraph(data, card_feature, merchant_feature):
    """
    Create a training graph from the transaction data.

    Parameters:
    data (pd.DataFrame): The original transaction data.
    card_feature (dict): Dictionary containing features for each card.
    merchant_feature (dict): Dictionary containing features for each merchant.

    Returns:
    Data: The created graph data for training, with node and edge features.
    """
    sample_data = downsample_data(data)
    graph, status = create_graph_from_sample(sample_data, card_feature, merchant_feature)
    while not status:
        sample_data = downsample_data(data)
        graph, status = create_graph_from_sample(sample_data, card_feature, merchant_feature)
    graph_data = from_networkx(graph)
    node_features = []
    for node in graph.nodes(data=True):
        node_attr = node[1]['features']
        node_features.append([
            node_attr['transaction_count'],
            node_attr['amount_mean'],
            node_attr['amount_std'],
            node_attr['time_diff_mean'],
            node_attr['time_diff_std']
        ])
    node_features = torch.tensor(node_features, dtype=torch.float)
    graph_data.x = node_features
    edge_features = []
    edge_labels = []
    for edge in graph.edges(data=True):
        edge_attr = edge[2]['features']
        edge_features.append([
            edge_attr['Year'],
            edge_attr['Month'],
            edge_attr['Day'],
            edge_attr['Time'],
            edge_attr['Amount'],
            edge_attr['Use Chip'],
            edge_attr['Merchant City'],
            edge_attr['Merchant State'],
            edge_attr['Zip'],
            edge_attr['MCC'],
            edge_attr['Errors?']
        ])
        edge_labels.append(edge_attr['Is Fraud?'])
    # print(edge_features)
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    graph_data.edge_attr = edge_features
    graph_data.edge_y = edge_labels
    return graph_data