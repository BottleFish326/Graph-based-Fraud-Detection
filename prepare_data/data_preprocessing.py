# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

def calculate_card_feature(data):
    """
    Calculate features for each card based on transaction data.

    Parameters:
    data (pd.DataFrame): The transaction data containing columns such as 'Card_id', 'Amount', 'Year', 'Month', 'Day', 'Time'.

    Returns:
    dict: A dictionary where each key is a card_id and the value is a dictionary of calculated features.
    """
    card_feature = {}
    for card_id, group in data.groupby('Card_id'):
        transaction_count = group.shape[0]
        group['Amount'] = group['Amount'].replace('[\$,]', '', regex=True).astype(float)
        amount_mean = group['Amount'].mean()
        amount_std = group['Amount'].std()

        group['DataTime'] = pd.to_datetime(group[['Year','Month','Day','Time']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H:%M')
        group = group.sort_values(by='DataTime')
        time_diffs = group['DataTime'].diff().dt.total_seconds().div(60).dropna()
        time_diff_mean = time_diffs.mean()
        time_diff_std = time_diffs.std()
        card_feature[card_id] = {
            'card_id': card_id,
            'transaction_count': transaction_count,
            'amount_mean': amount_mean,
            'amount_std': amount_std,
            'time_diff_mean': time_diff_mean,
            'time_diff_std': time_diff_std
        }
    return card_feature

def calculate_merchant_feature(data):
    """
    Calculate features for each merchant based on transaction data.

    Parameters:
    data (pd.DataFrame): The transaction data containing columns such as 'Merchant Name', 'Amount', 'Year', 'Month', 'Day', 'Time'.

    Returns:
    dict: A dictionary where each key is a merchant_name and the value is a dictionary of calculated features.
    """
    merchant_feature = {}
    for merchant_name, group in data.groupby('Merchant Name'):
        transaction_count = group.shape[0]
        group['Amount'] = group['Amount'].replace('[\$,]', '', regex=True).astype(float)
        amount_mean = group['Amount'].mean()
        amount_std = group['Amount'].std()

        group['DataTime'] = pd.to_datetime(group[['Year','Month','Day','Time']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H:%M')
        group = group.sort_values(by='DataTime')
        time_diffs = group['DataTime'].diff().dt.total_seconds().div(60).dropna()
        time_diff_mean = time_diffs.mean()
        time_diff_std = time_diffs.std()
        merchant_feature[merchant_name] = {
            'merchant_name': merchant_name,
            'transaction_count': transaction_count,
            'amount_mean': amount_mean,
            'amount_std': amount_std,
            'time_diff_mean': time_diff_mean,
            'time_diff_std': time_diff_std
        }
    return merchant_feature

def preprocess_data(data_path, data_preprocess_path, card_feature_path, merchant_feature_path):
    """
    Preprocess the credit card transaction data for fraud detection.

    Parameters:
    data_path (str): Path to the raw data file.
    data_preprocess_path (str): Path to save the preprocessed data.
    card_feature_path (str): Path to save the card feature data.
    merchant_feature_path (str): Path to save the merchant feature data.
    """
    data = read_data(data_path)
    data['Card_id'] = data.apply(lambda row: f'user_{row["User"]}_card_{row["Card"]}', axis=1)
    data['Is Fraud?'] = data['Is Fraud?'].replace({'Yes': 1, 'No': 0})
    le_merchant = LabelEncoder()
    le_cardid = LabelEncoder()
    le_chip = LabelEncoder()
    le_city = LabelEncoder()
    le_state = LabelEncoder()
    le_errors = LabelEncoder()
    data['Use Chip'] = le_chip.fit_transform(data['Use Chip'])
    data['Merchant City'] = le_city.fit_transform(data['Merchant City'])
    data['Merchant State'] = le_state.fit_transform(data['Merchant State'])
    data['Errors?'] = le_errors.fit_transform(data['Errors?'])
    data['Card_id'] = le_cardid.fit_transform(data['Card_id'])
    data['Merchant Name'] = le_merchant.fit_transform(data['Merchant Name'])
    data['Merchant Name'] = - data['Merchant Name']
    card_feature = calculate_card_feature(data)
    merchant_feature = calculate_merchant_feature(data)
    data['Amount'] = data['Amount'].replace('[\$,]', '', regex=True).astype(float)
    data['Time'] = data['Time'].str.replace(':', '').astype(int)
    scaler = MinMaxScaler()
    numeric_features = ['Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    data.fillna(0, inplace=True)
    data.to_csv(data_preprocess_path, index=False)
    df_card_feature = pd.DataFrame.from_dict(card_feature, orient='index')
    df_card_feature.fillna(0, inplace=True)
    df_card_feature.to_csv(card_feature_path, index=False)
    df_merchant_feature = pd.DataFrame.from_dict(merchant_feature, orient='index')
    df_merchant_feature.fillna(0, inplace=True)
    df_merchant_feature.to_csv(merchant_feature_path, index=False)

def initialize_data(data_path, card_feature_path, merchant_path):
    """
    Initialize and load the transaction data and features for cards and merchants.

    Parameters:
    data_path (str): The path to the transaction data file.
    card_feature_path (str): The path to the card feature data file.
    merchant_path (str): The path to the merchant feature data file.

    Returns:
    tuple: A tuple containing the transaction data (pd.DataFrame), card features (dict), and merchant features (dict).
    """
    data = read_data(data_path)
    card_feature = pd.read_csv(card_feature_path)
    card_feature.set_index('card_id', inplace=True)
    card_feature = card_feature.to_dict('index')
    merchant_feature = pd.read_csv(merchant_path)
    merchant_feature.set_index('merchant_name', inplace=True)
    merchant_feature = merchant_feature.to_dict('index')
    return data, card_feature, merchant_feature