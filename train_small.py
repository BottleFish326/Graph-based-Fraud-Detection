import torch
from prepare_data.data_preprocessing import preprocess_data
from prepare_data.data_preprocessing import initialize_data
from prepare_data.build_graph import create_traingraph
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from model.edgeGAT import edgeGAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def split_data(data, test_size=0.2):
    """
    Split the data into training and testing sets based on edges.

    Parameters:
    data (Data): The graph data containing edge_index and edge_attr.
    test_size (float): The proportion of data to be used for testing.

    Returns:
    train_data (Data): The training graph data.
    test_data (Data): The testing graph data.
    """
    edge_indices = list(range(data.edge_index.size(1)))
    train_indices, test_indices = train_test_split(edge_indices, test_size=test_size)

    train_edge_index = data.edge_index[:, train_indices]
    train_edge_attr = data.edge_attr[train_indices]
    train_labels = data.edge_y[train_indices]

    test_edge_index = data.edge_index[:, test_indices]
    test_edge_attr = data.edge_attr[test_indices]
    test_labels = data.edge_y[test_indices]

    train_data = Data(x=data.x, edge_index=train_edge_index, edge_attr=train_edge_attr, edge_y=train_labels)
    test_data = Data(x=data.x, edge_index=test_edge_index, edge_attr=test_edge_attr, edge_y=test_labels)

    return train_data, test_data

def train(train_loader):
    """
    Train the model using the provided data loader.

    Parameters:
    train_loader (DataLoader): The data loader for the training data.
    """
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.edge_y)
        loss.backward()
        optimizer.step()


def test(test_loader):
    """
    Test the model using the provided data loader and calculate accuracy and recall.

    Parameters:
    test_loader (DataLoader): The data loader for the testing data.

    Returns:
    accuracy (float): The accuracy of the model.
    recall (float): The recall of the model.
    """
    model.eval()
    correct = 0
    total = 0
    tp = 0
    fn = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.edge_y).sum().item()
        total += len(data.edge_y)
        tp += ((pred == 1) & (data.edge_y == 1)).sum().item()
        fn += ((pred == 0) & (data.edge_y == 1)).sum().item()
    return correct / total, tp / (tp + fn) if tp + fn != 0 else 0

preprocess_addr = './dataset/preprocessed_data.csv'
card_feature_addr = './dataset/card_feature.csv'
merchant_feature_addr = './dataset/merchant_feature.csv'

preprocess_data('./dataset/smalldata.csv', preprocess_addr, card_feature_addr, merchant_feature_addr)
data, card_feature, merchant_feature = initialize_data(preprocess_addr, card_feature_addr, merchant_feature_addr)
print("Start Training")

node_in_channels = 5
edge_in_channels = 11
out_channels = 16

model = edgeGAT(node_in_channels, edge_in_channels, out_channels, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for graph_num in range(10):
    graph_data = create_traingraph(data, card_feature, merchant_feature)
    train_data, test_data = split_data(graph_data)
    train_loader = DataLoader([train_data], batch_size=1)
    test_loader = DataLoader([test_data], batch_size=1)
    for epoch in range(1, 1001):
        train(train_loader)
        if epoch % 100 == 0:
            train_acc, train_recall = test(train_loader)
            test_acc, test_recall = test(test_loader)
            print(f'graph:{graph_num+1} Epoch:{epoch}, Train Accuarcy: {train_acc:.4f}, Train recall: {train_recall:.4f}, Test Accuracy: {test_acc:.4f}, Test recall: {test_recall:.4f}')