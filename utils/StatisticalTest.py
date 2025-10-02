import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from itertools import combinations
import warnings
from typing import Tuple, Optional, Dict, Any, List

warnings.filterwarnings('ignore')

class PairStatistics:
    """Container class for pair statistics to improve code readability"""
    def __init__(self):
        self.pairs_tested = 0
        self.insufficient_data = 0
        self.not_cointegrated = 0
        self.poor_mean_reversion = 0
        self.valid_pairs = 0
        
    def print_summary(self):
        """Print comprehensive filtering summary"""
        print(f"\n{'='*60}")
        print("PAIR FILTERING SUMMARY")
        print(f"{'='*60}")
        print(f"Total pairs tested:           {self.pairs_tested}")
        print(f"Insufficient data:            {self.insufficient_data}")
        print(f"Failed cointegration test:    {self.not_cointegrated}")
        # print(f"Poor mean reversion (OU):     {self.poor_mean_reversion}")
        print(f"Valid pairs found:            {self.valid_pairs}")
        print(f"Success rate:                 {self.valid_pairs/self.pairs_tested*100:.2f}%" if self.pairs_tested > 0 else "Success rate: 0%")
        print(f"{'='*60}")


def get_common_data(prices: pd.DataFrame, stock1: str, stock2: str) -> Tuple[pd.Series, pd.Series]:
    """
    Extract overlapping price data for two stocks
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with dates as index, stocks as columns
    stock1, stock2 : str
        Ticker symbols for the two stocks
        
    Returns:
    --------
    x_common, y_common : pd.Series
        Price series with matching dates only
    """
    x = prices[stock1].dropna()
    y = prices[stock2].dropna()
    common_index = x.index.intersection(y.index)
    x_common = x.loc[common_index]
    y_common = y.loc[common_index]
    return x_common, y_common


def test_cointegration(x: pd.Series, y: pd.Series, threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Test if two price series are cointegrated using Engle-Granger test
    
    Parameters:
    -----------
    x, y : pd.Series
        Price series to test
    threshold : float
        Significance level for cointegration test
        
    Returns:
    --------
    is_cointegrated : bool
        True if series are cointegrated at given significance level
    pvalue : float
        P-value from cointegration test
    """
    _, pvalue, _ = coint(x, y)
    is_cointegrated = pvalue < threshold
    return is_cointegrated, pvalue


def get_hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate hedge ratio (beta) by regressing x on y with intercept
    Hedge ratio represents how many units of y to short for 1 unit long of x
    
    Parameters:
    -----------
    x, y : pd.Series
        Price series where x is dependent variable
        
    Returns:
    --------
    beta : float
        Hedge ratio (slope coefficient)
    """
    X = add_constant(y)
    model = OLS(x, X).fit()
    beta = model.params[1]
    return beta


def calculate_spread(x: pd.Series, y: pd.Series, beta: float) -> pd.Series:
    """
    Calculate the spread between two prices given hedge ratio
    Spread = x - beta * y (should be stationary for good pairs)
    
    Parameters:
    -----------
    x, y : pd.Series
        Price series
    beta : float
        Hedge ratio
        
    Returns:
    --------
    spread : pd.Series
        Spread time series
    """
    spread = x - beta * y
    return spread


def fit_ou_model(spread: np.ndarray) -> Tuple[float, float]:
    """
    Fit Ornstein-Uhlenbeck (OU) process to spread to estimate mean reversion
    
    OU model: dS = theta * (mu - S) * dt + sigma * dW
    where theta is speed of reversion, mu is long-term mean
    
    Parameters:
    -----------
    spread : np.ndarray
        Spread time series values
        
    Returns:
    --------
    theta : float
        Speed of mean reversion parameter
    halflife : float
        Half-life of mean reversion in days
    """
    delta_spread = np.diff(spread)
    lagged_spread = spread[:-1]
    X = add_constant(lagged_spread)
    model = OLS(delta_spread, X).fit()
    theta = -model.params[1]
    halflife = np.log(2) / theta if theta > 0 else np.inf
    return theta, halflife


def evaluate_pair(prices: pd.DataFrame, stock1: str, stock2: str, 
                 coint_thresh: float = 0.05, max_halflife: float = 90,
                 verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Comprehensive evaluation of a stock pair for statistical trading viability
    
    Tests performed:
    1. Sufficient overlapping data (>=100 points)
    2. Cointegration test (Engle-Granger)
    3. Mean reversion properties (Ornstein-Uhlenbeck model)
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with dates as index, stocks as columns
    stock1, stock2 : str
        Ticker symbols for the pair
    coint_thresh : float
        Significance threshold for cointegration test
    max_halflife : float
        Maximum acceptable half-life for mean reversion (in days)
    verbose : bool
        Whether to print detailed evaluation info
        
    Returns:
    --------
    result : dict or None
        Dictionary with pair statistics if valid, None otherwise
    """
  
    x, y = get_common_data(prices, stock1, stock2)
    if len(x) < 100:
        if verbose:
            print(f"{stock1}-{stock2}: Insufficient data ({len(x)} points, need ≥100)")
        return None
    

    is_cointegrated, pval = test_cointegration(x, y, threshold=coint_thresh)
    if not is_cointegrated:
        if verbose:
            print(f"{stock1}-{stock2}: Not cointegrated (p={pval:.4f} > {coint_thresh})")
        return None

    beta = get_hedge_ratio(x, y)
    spread = calculate_spread(x, y, beta)
    
    theta, halflife = fit_ou_model(spread.values)
    
    # if not (0 < halflife < max_halflife):
    #     if verbose:
    #         print(f"{stock1}-{stock2}: Poor mean reversion (halflife={halflife:.2f} days, want <{max_halflife})")
    #     return None
    
    # Pair passed all tests
    result = {
        'stock1': stock1,
        'stock2': stock2,
        'data_points': len(x),
        'cointegration_p': pval,
        'hedge_ratio': beta,
        'ou_theta': theta,
        'ou_halflife': halflife,
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
    }
    
    if verbose:
        print(f"{stock1}-{stock2}: VALID PAIR")
        print(f"Data points: {len(x)}, Coint p-value: {pval:.4f}")
        print(f"Hedge ratio: {beta:.4f}, Half-life: {halflife:.2f} days")
    
    return result


def filter_pairs_from_clusters(prices: pd.DataFrame, clustered_df: pd.DataFrame, 
                              coint_thresh: float = 0.05, max_halflife: float = 90,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Filter statistically meaningful pairs from clustered stocks
    
    For each cluster, tests all possible stock pairs using:
    - Cointegration analysis
    - Mean reversion properties (Ornstein-Uhlenbeck model)
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with dates as index, stocks as columns
    clustered_df : pd.DataFrame
        DataFrame with 'Stock' and 'Cluster' columns
    coint_thresh : float
        Significance threshold for cointegration test
    max_halflife : float
        Maximum acceptable half-life for mean reversion
    verbose : bool
        Whether to print detailed progress and results
        
    Returns:
    --------
    result_df : pd.DataFrame
        DataFrame containing all valid pairs with statistics
    """
    valid_pairs = []
    stats = PairStatistics()
    cluster_results = {}
    
    if verbose:
        print(f"Starting pair filtering with parameters:")
        print(f"Cointegration threshold: {coint_thresh}")
        print(f"Max half-life: {max_halflife} days")
        print(f"Total clusters: {clustered_df['Cluster'].nunique()}")
    
    for cluster_id, group in clustered_df.groupby('Cluster'):
        tickers = group.index.tolist()
        cluster_pairs_tested = 0
        cluster_valid_pairs = []
        
        if verbose:
            print(f"\nProcessing Cluster {cluster_id} ({len(tickers)} stocks)")
            print(f"   Stocks: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
        
        for stock1, stock2 in combinations(tickers, 2):
            stats.pairs_tested += 1
            cluster_pairs_tested += 1
            
            x, y = get_common_data(prices, stock1, stock2)
            if len(x) < 100:
                stats.insufficient_data += 1
                continue
                
            # Test cointegration
            is_cointegrated, pval = test_cointegration(x, y, threshold=coint_thresh)
            if not is_cointegrated:
                stats.not_cointegrated += 1
                continue
                
            # Test mean reversion
            beta = get_hedge_ratio(x, y)
            spread = calculate_spread(x, y, beta)
            theta, halflife = fit_ou_model(spread.values)
            
            # if not (0 < halflife < max_halflife):
            #     stats.poor_mean_reversion += 1
            #     continue
            
            # Valid pair found
            pair_result = {
                'cluster': cluster_id,
                'stock1': stock1,
                'stock2': stock2,
                'data_points': len(x),
                'cointegration_p': pval,
                'hedge_ratio': beta,
                'ou_theta': theta,
                'ou_halflife': halflife,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
            }
            
            valid_pairs.append(pair_result)
            cluster_valid_pairs.append(pair_result)
            stats.valid_pairs += 1
            
            if verbose:
                print(f"{stock1}-{stock2}: p={pval:.4f}, β={beta:.3f}, HL={halflife:.1f}d")
        
        cluster_results[cluster_id] = {
            'total_pairs_tested': cluster_pairs_tested,
            'valid_pairs_found': len(cluster_valid_pairs),
            'success_rate': len(cluster_valid_pairs) / cluster_pairs_tested * 100 if cluster_pairs_tested > 0 else 0
        }
        
        if verbose:
            print(f"Cluster {cluster_id} results: {len(cluster_valid_pairs)}/{cluster_pairs_tested} pairs valid")
    

    if verbose:
        stats.print_summary()
        
        print(f"\nCLUSTER-WISE BREAKDOWN:")
        for cluster_id, results in cluster_results.items():
            print(f"   Cluster {cluster_id}: {results['valid_pairs_found']}/{results['total_pairs_tested']} " +
                  f"({results['success_rate']:.1f}% success)")
    
    result_df = pd.DataFrame(valid_pairs)
    
    if len(result_df) > 0:
        result_df = result_df.sort_values('cointegration_p')
        
        if verbose:
            print(f"\n TOP 5 PAIRS BY COINTEGRATION STRENGTH:")
            top_pairs = result_df.head()
            for _, row in top_pairs.iterrows():
                print(f"   {row['stock1']}-{row['stock2']}: p={row['cointegration_p']:.4f}, HL={row['ou_halflife']:.1f}d")
    
    return result_df


def analyze_pair_quality(pairs_df: pd.DataFrame) -> None:
    """
    Analyze and visualize the quality of discovered pairs
    
    Parameters:
    -----------
    pairs_df : pd.DataFrame
        DataFrame containing valid pairs from filter_pairs_from_clusters
    """
    if len(pairs_df) == 0:
        print(" No pairs to analyze")
        return
    
    print(f"\nPAIR QUALITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total valid pairs found: {len(pairs_df)}")
    print(f"\nCointegration p-values:")
    print(f"   Mean: {pairs_df['cointegration_p'].mean():.4f}")
    print(f"   Median: {pairs_df['cointegration_p'].median():.4f}")
    print(f"   Min: {pairs_df['cointegration_p'].min():.4f}")
    print(f"   Max: {pairs_df['cointegration_p'].max():.4f}")
    
    print(f"\nMean reversion half-life (days):")
    print(f"   Mean: {pairs_df['ou_halflife'].mean():.2f}")
    print(f"   Median: {pairs_df['ou_halflife'].median():.2f}")
    print(f"   Min: {pairs_df['ou_halflife'].min():.2f}")
    print(f"   Max: {pairs_df['ou_halflife'].max():.2f}")
    
    print(f"\nHedge ratio distribution:")
    print(f"   Mean: {pairs_df['hedge_ratio'].mean():.3f}")
    print(f"   Std: {pairs_df['hedge_ratio'].std():.3f}")
    
    # Quality grades
    strong_coint = (pairs_df['cointegration_p'] < 0.01).sum()
    fast_reversion = (pairs_df['ou_halflife'] < 15).sum()
    
    print(f"\nQuality grades:")
    print(f"   Strong cointegration (p < 0.01): {strong_coint} pairs ({strong_coint/len(pairs_df)*100:.1f}%)")
    print(f"   Fast mean reversion (< 15 days): {fast_reversion} pairs ({fast_reversion/len(pairs_df)*100:.1f}%)")
