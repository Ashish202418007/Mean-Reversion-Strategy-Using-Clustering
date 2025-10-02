import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from pathlib import Path

class StockClustering:
    """
    A class to perform and visualize hierarchical clustering on stock correlation data.
    """
    def __init__(self, partial_corr, output_dir, max_clusters=40):
        self.partial_corr_matrix = partial_corr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_clusters = max_clusters
        
        self.distance_matrix = None
        self.linkage_matrix = None
        self.hierarchical_results = None

    def create_distance_matrix(self):
        """
        Converts the partial correlation matrix to a distance matrix
        using the formula: d = sqrt(1 - rho).
        """
        distance_matrix_vals = np.sqrt(1 - self.partial_corr_matrix.values)
        distance_matrix_vals = (distance_matrix_vals + distance_matrix_vals.T) / 2

        np.fill_diagonal(distance_matrix_vals, 0)
        self.distance_matrix = pd.DataFrame(
            distance_matrix_vals,
            index=self.partial_corr_matrix.index,
            columns=self.partial_corr_matrix.columns   
        )
        return self.distance_matrix

    def _find_optimal_clusters(self, plot=True):
        """
        Finds the optimal number of clusters using the silhouette score.
        """
        silhouette_scores = []
        cluster_range = range(2, self.max_clusters + 1)
        condensed_distances = squareform(self.distance_matrix.values)
        self.linkage_matrix = linkage(condensed_distances, method='ward')
        
        for n_clusters in cluster_range:
            cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            silhouette_avg = silhouette_score(self.distance_matrix.values, cluster_labels, metric='precomputed')
            silhouette_scores.append(silhouette_avg)

        optimal_n = cluster_range[np.argmax(silhouette_scores)]
        
        if plot:
            self._plot_silhouette_analysis(cluster_range, silhouette_scores, optimal_n)
        
        print(f"Optimal number of clusters found: {optimal_n}")
        return optimal_n

    def _plot_silhouette_analysis(self, cluster_range, scores, optimal_n):
        """Plots the silhouette scores to visualize the optimal cluster count."""
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, scores, 'bo-', linewidth=2, markersize=8)
        plt.title('Silhouette Analysis for Optimal Cluster Count')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Average Silhouette Score')
        plt.axvline(x=optimal_n, color='red', linestyle='--', label=f'Optimal: {optimal_n} clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = self.output_dir / "silhouette_analysis.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def perform_hierarchical_clustering(self, n_clusters):
        """
        Performs clustering with the specified number of clusters and saves the results.
        """
        cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        self.hierarchical_results = pd.DataFrame({
            'Ticker': self.distance_matrix.index,
            'Cluster': cluster_labels
        })
        path = self.output_dir / "hierarchical_cluster_assignments.csv"
        self.hierarchical_results.to_csv(path, index=False)


    def plot_dendrogram(self, **kwargs):
        """
        Generates and saves a dendrogram visualization of the clustering.
        """
        print("Generating dendrogram visualization...")
        plt.figure(figsize=(20, 10))
        plt.title('Hierarchical Clustering Dendrogram of NIFTY 50 Stocks')
        plt.xlabel('Stock Ticker')
        plt.ylabel('Distance')
        dendrogram(
            self.linkage_matrix, 
            labels=self.distance_matrix.index.tolist(), 
            leaf_rotation=90, 
            leaf_font_size=8,
            **kwargs
        )
        path = self.output_dir / "dendrogram.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dendrogram saved to {path}")
        
    def run_pipeline(self, create_visualizations=True):
        """
        Executes the complete clustering pipeline from start to finish.
        """
        self.create_distance_matrix()
        optimal_n_clusters = self._find_optimal_clusters(plot=create_visualizations)
        self.perform_hierarchical_clustering(n_clusters=optimal_n_clusters)
        
        if create_visualizations:
            self.plot_dendrogram()
    
        return self.hierarchical_results


