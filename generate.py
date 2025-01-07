import torch
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import networkx as nx

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Number of CUDA Devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


data = pd.read_csv('./dataset/data.csv')
fraud_data = data[data['Is Fraud?'] == "Yes"]
non_fraud_data = data[data['Is Fraud?'] == "No"]
non_fraud_data = resample(non_fraud_data, 
                                replace=False, 
                                n_samples=len(fraud_data)*10, 
                                random_state=42)
sampled_data = pd.concat([fraud_data, non_fraud_data])
sampled_data.to_csv('./dataset/smalldata.csv')
print(data.head())