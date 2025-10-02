"""Execute the complete clustering and validation pipeline using hierarchical clustering."""

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import warnings
import requests

from src.data_collection import fetch_multiple_tickers_parallel
from src.preprocess import preprocess_stock_data
from utils.clustering import StockClustering
from utils.StatisticalTest import filter_pairs_from_clusters
from utils.validator import ClusteringRobustnessValidator


warnings.filterwarnings('ignore')


load_dotenv("./.env")
TICKER_URL = os.getenv('NIFTY50_URL')

CONFIG = {
    'VALIDATION_DIR' : './pipeline/',
    'START_DATE': '2016-01-01',
    'END_DATE': '2024-12-31',
    'OUTPUT_DIR': './pipeline',
    'MAX_CLUSTERS': 6,
    'N_BOOTSTRAP': 1000,
    'N_MONTE_CARLO': 1000,
    'MATRIX_PATH': './data/processed/partial_corr_matrix.csv'
}

print("Starting Stock Clustering and Robustness Validation Pipeline")
print("=" * 60)

it_sector_tickers = [
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
        'LTIM.NS', 'MPHASIS.NS', 'OFSS.NS', 'PERSISTENT.NS', 'COFORGE.NS'
    ]

banking_sector_tickers = [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
        'AXISBANK.NS', 'INDUSINDBK.NS', 'BANKBARODA.NS'
    ]

def download(ticker_list, start_date, end_date, output_path):
    data = fetch_multiple_tickers_parallel(
        ticker_list=ticker_list, start=start_date, end=end_date, interval='1d', timeout=60)
    data.to_csv(output_path, index=False)
    return data

def adaptive_train_test_split(df):
    dates = pd.to_datetime(df['Date'].unique())
    dates = np.sort(dates)
    split_idx = pd.to_datetime('2018-12-31 00:00:00+05:30')

    train_data = df[df['Date']<=(split_idx)]
    test_data = df[df['Date']>(split_idx)]
    return train_data, test_data

def main():
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36"
    }

    response = requests.get(TICKER_URL, headers=headers)
    response.raise_for_status() 

    tables = pd.read_html(response.text)
    tickers = tables[1]['Symbol'].tolist()
    print(tickers[:10])
    tickers = [t + '.NS' for t in tickers]
    if 'ETERNAL.NS' in tickers:
        tickers[tickers.index('ETERNAL.NS')] = 'ZOMATO.NS'

    OUTPUT_DIR = CONFIG['OUTPUT_DIR']
    VALIDATION_DIR = CONFIG['VALIDATION_DIR']
 
    csv_path = os.path.join(OUTPUT_DIR, f"NIFTY50_{CONFIG['START_DATE']}_{CONFIG['END_DATE']}.csv")
    data = download(banking_sector_tickers,CONFIG['START_DATE'], CONFIG['END_DATE'], csv_path)
    print(f'Downloaded {len(data)} records.')


    train_data, test_data = adaptive_train_test_split(data)
    print(f'Train records: {len(train_data)}, Test records: {len(test_data)}')

    train_data.to_csv(os.path.join(VALIDATION_DIR, 'train.csv'))

    returns_pivot_train, partial_corr_matrix = preprocess_stock_data(os.path.join(VALIDATION_DIR, 'train.csv'))
    clustering = StockClustering(partial_corr=partial_corr_matrix,  
                                output_dir=VALIDATION_DIR,
                                max_clusters=CONFIG['MAX_CLUSTERS'])
    
    cluster_results_train = clustering.run_pipeline(create_visualizations=True)
    print(cluster_results_train.head())

    cluster_labels_train = cluster_results_train['Cluster'].values
    tickers_train = returns_pivot_train.columns.tolist()


    test_returns_pivot = test_data.pivot(index='Date', columns='Ticker', values='Close').pct_change().dropna()
    common_tickers = [t for t in tickers_train if t in test_returns_pivot.columns]
    train_mat = returns_pivot_train[common_tickers].values
    test_mat = test_returns_pivot[common_tickers].values
    cluster_labels_train = cluster_results_train.set_index('Ticker').loc[common_tickers, 'Cluster'].values

    validator = ClusteringRobustnessValidator(random_state=42)
    # Bootstrap 
    bootstrap_results = validator.bootstrap_stability_test(train_mat, cluster_labels_train, n_bootstrap=CONFIG['N_BOOTSTRAP'])
    # Monte Carlo 
    monte_carlo_results = validator.monte_carlo_significance_test(n_simulations=CONFIG['N_MONTE_CARLO'])
    # Out of Sample 
    temporal_results = validator.temporal_validation_test(
        train_data=train_mat, test_data=test_mat, original_labels=cluster_labels_train)


    print("Bootstrap Stability:", bootstrap_results['stability_score'])
    print("Monte Carlo p-value:", monte_carlo_results['modularity_p_value'])
    print("Temporal ARI out-of-sample:", temporal_results['temporal_ari'])
    print("Cointegration rate out-of-sample:", temporal_results['cointegration_rate'])

if __name__ == "__main__":
    main()
