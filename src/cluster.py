from utils.clustering import StockClustering
import pandas as pd

MATRIX_PATH = r".\data\processed\PC_matrix.csv"
OUTPUT_DIR = r".\output"

try:
    partial_corr_matrix = pd.read_csv(MATRIX_PATH, index_col=0)
except FileNotFoundError:
    print(f"Error: The file '{MATRIX_PATH}' was not found.")
    print("Please ensure your partial correlation matrix is in the correct directory.")
    exit()

clustering_analysis = StockClustering(
    partial_corr=partial_corr_matrix,
    output_dir=OUTPUT_DIR,
    max_clusters=40
)

results_df = clustering_analysis.run_pipeline(create_visualizations=True)

# summary
print("\n--- Clustering Results ---")
print(results_df.head())

print("\n--- Cluster Sizes ---")
print(results_df['Cluster'].value_counts().sort_index())