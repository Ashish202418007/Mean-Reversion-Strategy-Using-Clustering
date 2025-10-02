from utils.StatisticalTest import filter_pairs_from_clusters
import pandas as pd
from pathlib import Path

DATA = r".\data\raw\Nifty50data.csv"
HIERARCHICAL = r".\output\hierarchical_cluster_assignments.csv"
OUTPUT = r".\res\pairs_hierarchical.csv"

prices = pd.read_csv(DATA)
prices = prices.pivot(index='Date', columns='Ticker', values='Close')
print(prices.head(2))

hier_clusters = pd.read_csv(HIERARCHICAL).set_index('Stock')

pairs_hier = filter_pairs_from_clusters(prices, hier_clusters)
pairs_hier.to_csv(OUTPUT, index=False)
