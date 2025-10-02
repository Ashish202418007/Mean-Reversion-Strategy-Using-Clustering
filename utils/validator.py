import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.tsa.stattools import coint
from typing import Dict, Any
from pathlib import Path
from utils.clustering import StockClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')


class ClusteringRobustnessValidator:
    """
    A comprehensive pipeline for testing the statistical robustness of clustering methods
    specifically designed for financial time series and hierarchical clustering.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        self.original_clustering = None
        self.original_data = None
        self.original_corr_matrix = None
        self.clustering_method = 'hierarchical'  
        
    def compute_partial_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix from time series data."""
        df = pd.DataFrame(data)
        corr_matrix = df.corr().values
        
        try:
            # Use regularized inverse for numerical stability
            reg_param = 1e-6
            regularized_corr = corr_matrix + reg_param * np.eye(corr_matrix.shape[0])
            precision_matrix = np.linalg.inv(regularized_corr)
            
            # Convert precision matrix to partial correlation
            partial_corr = np.zeros_like(precision_matrix)
            for i in range(len(precision_matrix)):
                for j in range(len(precision_matrix)):
                    if i != j:
                        partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(
                            precision_matrix[i, i] * precision_matrix[j, j]
                        )
                    else:
                        partial_corr[i, j] = 1.0
        except np.linalg.LinAlgError:
            print("Warning: Partial correlation computation failed, using regular correlation")
            partial_corr = corr_matrix
            
        return partial_corr
    
    def perform_hierarchical_clustering_on_matrix(self, correlation_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform hierarchical clustering on a correlation matrix."""
        # Convert correlation to distance
        clustering = StockClustering(pd.DataFrame(correlation_matrix), output_dir=".", max_clusters=n_clusters)
        cluster_results_train = clustering.run_pipeline(create_visualizations=False)
        
        cluster_labels = cluster_results_train['Cluster'].values
        
        return cluster_labels
    
    def compute_modularity(self, similarity_matrix: np.ndarray, labels: np.ndarray) -> float:
        """Compute modularity score for a clustering solution."""
        adj_matrix = np.abs(similarity_matrix)
        threshold = np.percentile(adj_matrix[adj_matrix > 0], 75)
        adj_matrix = (adj_matrix > threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        
        G = nx.from_numpy_array(adj_matrix)
        
        unique_labels = np.unique(labels)
        communities = []
        for label in unique_labels:
            community = np.where(labels == label)[0].tolist()
            if len(community) > 0:
                communities.append(community)
        
        try:
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = 0.0
            
        return modularity
    
    def generate_null_model(self, original_matrix: np.ndarray) -> np.ndarray:
        """Generate a null model matrix that preserves degree distribution."""
        n = original_matrix.shape[0]
        strengths = np.sum(np.abs(original_matrix), axis=1)
        
        null_matrix = np.random.randn(n, n)
        null_matrix = (null_matrix + null_matrix.T) / 2
        np.fill_diagonal(null_matrix, 1.0)
        
        current_strengths = np.sum(np.abs(null_matrix), axis=1)
        for i in range(n):
            if current_strengths[i] > 0:
                scale_factor = strengths[i] / current_strengths[i]
                null_matrix[i, :] *= scale_factor
                null_matrix[:, i] *= scale_factor
        
        null_matrix = np.clip(null_matrix, -0.99, 0.99)
        np.fill_diagonal(null_matrix, 1.0)
        
        return null_matrix

    def bootstrap_stability_test(self, data: np.ndarray, original_labels: np.ndarray, 
                                n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap resampling test for clustering stability using hierarchical clustering."""
        print(f"Running Bootstrap Stability Test with {n_bootstrap} iterations...")
        
        self.original_data = data.copy()
        original_corr_matrix = self.compute_partial_correlation(data)
        self.original_corr_matrix = original_corr_matrix
        self.original_clustering = original_labels
        
        original_silhouette = silhouette_score(original_corr_matrix, original_labels)
        original_n_clusters = len(np.unique(original_labels))
        
        ari_scores = []
        silhouette_scores = []
        n_clusters_list = []
        
        for i in range(n_bootstrap):
            if i % 100 == 0:
                print(f"  Bootstrap iteration {i}/{n_bootstrap}")
            
            n_timestamps = data.shape[0]
            bootstrap_indices = np.random.choice(n_timestamps, size=n_timestamps, replace=True)
            bootstrap_data = data[bootstrap_indices]
            
            bootstrap_corr = self.compute_partial_correlation(bootstrap_data)
            
            try:
                bootstrap_labels = self.perform_hierarchical_clustering_on_matrix(
                    bootstrap_corr, original_n_clusters)
                
                ari = adjusted_rand_score(original_labels, bootstrap_labels)
                silhouette = silhouette_score(bootstrap_corr, bootstrap_labels)
                n_clusters = len(np.unique(bootstrap_labels))
                
                ari_scores.append(ari)
                silhouette_scores.append(silhouette)
                n_clusters_list.append(n_clusters)
                
            except Exception as e:
                print(e)
        
        bootstrap_results = {
            'ari_scores': np.array(ari_scores),
            'ari_mean': np.mean(ari_scores),
            'ari_std': np.std(ari_scores),
            'ari_median': np.median(ari_scores),
            'ari_q25': np.percentile(ari_scores, 25),
            'ari_q75': np.percentile(ari_scores, 75),
            'silhouette_scores': np.array(silhouette_scores),
            'silhouette_mean': np.mean(silhouette_scores),
            'silhouette_std': np.std(silhouette_scores),
            'n_clusters_list': np.array(n_clusters_list),
            'n_clusters_mean': np.mean(n_clusters_list),
            'n_clusters_std': np.std(n_clusters_list),
            'original_silhouette': original_silhouette,
            'original_n_clusters': original_n_clusters,
            'n_successful_iterations': len(ari_scores),
            'stability_score': np.mean(ari_scores)
        }
        
        self.results['bootstrap'] = bootstrap_results
        
        print(f"Bootstrap test completed!")
        print(f"  Mean ARI: {bootstrap_results['ari_mean']:.4f} ± {bootstrap_results['ari_std']:.4f}")
        print(f"  Stability Score: {bootstrap_results['stability_score']:.4f}")
        
        return bootstrap_results

    def monte_carlo_significance_test(self, n_simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo simulation to test statistical significance of clustering."""
        print(f"Running Monte Carlo Significance Test with {n_simulations} simulations...")
        
        if self.original_corr_matrix is None or self.original_clustering is None:
            raise ValueError("Must run bootstrap test first to establish original clustering")
        
        original_modularity = self.compute_modularity(self.original_corr_matrix, self.original_clustering)
        original_silhouette = silhouette_score(self.original_corr_matrix, self.original_clustering)
        original_n_clusters = len(np.unique(self.original_clustering))
        
        null_modularities = []
        null_silhouettes = []
        
        for i in range(n_simulations):
            if i % 100 == 0:
                print(f"  Monte Carlo iteration {i}/{n_simulations}")
            
            null_matrix = self.generate_null_model(self.original_corr_matrix)
            
            try:
                null_labels = self.perform_hierarchical_clustering_on_matrix(null_matrix, original_n_clusters)
                
                null_modularity = self.compute_modularity(null_matrix, null_labels)
                null_silhouette = silhouette_score(null_matrix, null_labels)
                
                null_modularities.append(null_modularity)
                null_silhouettes.append(null_silhouette)
                
            except Exception as e:
                continue
        
        null_modularities = np.array(null_modularities)
        null_silhouettes = np.array(null_silhouettes)
        
        modularity_p_value = np.mean(null_modularities >= original_modularity)
        silhouette_p_value = np.mean(null_silhouettes >= original_silhouette)
        
        monte_carlo_results = {
            'original_modularity': original_modularity,
            'original_silhouette': original_silhouette,
            'null_modularities': null_modularities,
            'null_modularity_mean': np.mean(null_modularities),
            'null_modularity_std': np.std(null_modularities),
            'null_silhouettes': null_silhouettes,
            'null_silhouette_mean': np.mean(null_silhouettes),
            'null_silhouette_std': np.std(null_silhouettes),
            'modularity_p_value': modularity_p_value,
            'silhouette_p_value': silhouette_p_value,
            'modularity_z_score': (original_modularity - np.mean(null_modularities)) / np.std(null_modularities) if np.std(null_modularities) > 0 else 0,
            'silhouette_z_score': (original_silhouette - np.mean(null_silhouettes)) / np.std(null_silhouettes) if np.std(null_silhouettes) > 0 else 0,
            'n_successful_simulations': len(null_modularities),
            'is_significant_modularity': modularity_p_value < 0.05,
            'is_significant_silhouette': silhouette_p_value < 0.05
        }
        
        self.results['monte_carlo'] = monte_carlo_results
        
        print(f"Monte Carlo test completed!")
        print(f"  Original Modularity: {original_modularity:.4f}")
        print(f"  Null Modularity Mean: {np.mean(null_modularities):.4f} ± {np.std(null_modularities):.4f}")
        print(f"  Modularity p-value: {modularity_p_value:.6f}")
        print(f"  Significant: {modularity_p_value < 0.05}")
        
        return monte_carlo_results

    def temporal_validation_test(self, train_data, test_data, original_labels: np.ndarray, stock_names=None) -> Dict[str, Any]:
        """
        Temporal validation: check if cluster structure holds in test data.
        """

        original_n_clusters = len(np.unique(original_labels))
        test_data_corr = self.compute_partial_correlation(test_data)
        test_labels = self.perform_hierarchical_clustering_on_matrix(test_data_corr, original_n_clusters)

        # Caution: ARI assumes same dataset, here we interpret it as "stability of clustering across time"
        temporal_ari = adjusted_rand_score(original_labels, test_labels)

        cointegration_results = []
        for label in np.unique(original_labels):
            cluster_indices = np.where(original_labels == label)[0]
            if len(cluster_indices) < 2:
                continue

            for stock_i, stock_j in combinations(cluster_indices, 2):
                series_i = test_data[:, stock_i]
                series_j = test_data[:, stock_j]

                try:
                    _, p_value, _ = coint(series_i, series_j)
                    is_cointegrated = p_value < 0.05  # TODO: adjust for multiple testing

                    stock_pair = (stock_names[stock_i], stock_names[stock_j]) if stock_names else (stock_i, stock_j)
                    cointegration_results.append({
                        'stock_pair': stock_pair,
                        'cluster': label,
                        'p_value': p_value,
                        'is_cointegrated': is_cointegrated
                    })
                except Exception:
                    continue

        total_pairs = len(cointegration_results)
        cointegrated_pairs = sum(r['is_cointegrated'] for r in cointegration_results)
        cointegration_rate = cointegrated_pairs / total_pairs if total_pairs > 0 else 0

        results = {
            'temporal_ari': temporal_ari,
            'train_n_clusters': original_n_clusters,
            'test_n_clusters': len(np.unique(test_labels)),
            'cointegration_results': cointegration_results,
            'total_pairs_tested': total_pairs,
            'cointegrated_pairs': cointegrated_pairs,
            'cointegration_rate': cointegration_rate
        }

        print("Temporal validation completed!")
        print(f"  Temporal ARI: {temporal_ari:.4f}")
        print(f"  Cointegration Rate: {cointegration_rate:.4f} ({cointegrated_pairs}/{total_pairs} pairs)")

        return results



    # def run_full_validation(self, data: np.ndarray, original_labels: np.ndarray, 
    #                        n_bootstrap: int = 1000, n_monte_carlo: int = 1000, 
    #                        train_ratio: float = 0.7) -> Dict[str, Any]:
    #     """Run all three validation tests in sequence using hierarchical clustering."""
    #     print("=== STARTING COMPREHENSIVE CLUSTERING ROBUSTNESS VALIDATION ===\n")
        
    #     # Step 1: Bootstrap Stability Test
    #     bootstrap_results = self.bootstrap_stability_test(data, original_labels, n_bootstrap)
        
    #     print("\n" + "="*60 + "\n")
        
    #     # Step 2: Monte Carlo Significance Test
    #     monte_carlo_results = self.monte_carlo_significance_test(n_monte_carlo)
        
    #     print("\n" + "="*60 + "\n")
        
    #     # Step 3: Temporal Validation Test
    #     temporal_results = self.temporal_validation_test(data, train_ratio)
        
    #     print("\n=== VALIDATION SUMMARY ===")
    #     print(f"Bootstrap Stability Score: {bootstrap_results['stability_score']:.4f}")
    #     print(f"Monte Carlo p-value (Modularity): {monte_carlo_results['modularity_p_value']:.6f}")
    #     print(f"Temporal Stability (ARI): {temporal_results['temporal_ari']:.4f}")
    #     print(f"Cointegration Success Rate: {temporal_results['cointegration_rate']:.4f}")
        
    #     # Overall assessment
    #     is_robust = (
    #         bootstrap_results['stability_score'] > 0.7 and
    #         monte_carlo_results['modularity_p_value'] < 0.05 and
    #         temporal_results['temporal_ari'] > 0.5 and
    #         temporal_results['cointegration_rate'] > 0.3
    #     )
        
    #     print(f"\nOVERALL ASSESSMENT: {'ROBUST' if is_robust else 'NEEDS IMPROVEMENT'}")
        
    #     return {
    #         'bootstrap': bootstrap_results,
    #         'monte_carlo': monte_carlo_results,
    #         'temporal': temporal_results,
    #         'is_robust': is_robust
    #     }

    # def save_results_to_csv(self, filepath: str = "clustering_validation_results.csv"):
    #     """Save all validation results to CSV files."""
    #     base_path = Path(filepath).parent
        
    #     if 'bootstrap' in self.results:
    #         bootstrap_df = pd.DataFrame({
    #             'metric': ['ari_mean', 'ari_std', 'ari_median', 'stability_score', 'original_silhouette'],
    #             'value': [
    #                 self.results['bootstrap']['ari_mean'],
    #                 self.results['bootstrap']['ari_std'],
    #                 self.results['bootstrap']['ari_median'],
    #                 self.results['bootstrap']['stability_score'],
    #                 self.results['bootstrap']['original_silhouette']
    #             ]
    #         })
    #         bootstrap_df.to_csv(base_path / 'bootstrap_results.csv', index=False)
        
    #     if 'monte_carlo' in self.results:
    #         mc_df = pd.DataFrame({
    #             'metric': ['original_modularity', 'null_modularity_mean', 'modularity_p_value', 'modularity_z_score'],
    #             'value': [
    #                 self.results['monte_carlo']['original_modularity'],
    #                 self.results['monte_carlo']['null_modularity_mean'],
    #                 self.results['monte_carlo']['modularity_p_value'],
    #                 self.results['monte_carlo']['modularity_z_score']
    #             ]
    #         })
    #         mc_df.to_csv(base_path / 'monte_carlo_results.csv', index=False)
        
    #     if 'temporal' in self.results:
    #         temporal_df = pd.DataFrame(self.results['temporal']['cointegration_results'])
    #         temporal_df.to_csv(base_path / 'temporal_cointegration_results.csv', index=False)
        
    #     print("Results saved to CSV files!")


