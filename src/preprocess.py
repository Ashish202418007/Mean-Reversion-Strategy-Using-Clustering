import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


STOCK = r".\data\raw\Nifty50data.csv"
OUTPUT = Path(r".\data\processed")

def calculate_individual_features(df, window=21):
    """
    Calculate features directly from the original DataFrame.
    Much simpler - no pivoting headaches!
    """
    features = {}
    
    for ticker in df['Ticker'].unique():
        stock_data = df[df['Ticker'] == ticker].set_index('Date').sort_index()
        
        returns = stock_data['Close'].pct_change()
        features[f'{ticker}_returns'] = returns
        features[f'{ticker}_volatility'] = returns.rolling(window).std()
        features[f'{ticker}_momentum'] = stock_data['Close'].pct_change(5)
        features[f'{ticker}_price_zscore'] = ((stock_data['Close'] - stock_data['Close'].rolling(252).mean()) / 
                                            stock_data['Close'].rolling(252).std())

        features[f'{ticker}_volume_ratio'] = stock_data['Volume'] / stock_data['Volume'].rolling(21).mean()
        features[f'{ticker}_volume_momentum'] = stock_data['Volume'].pct_change(5)
        features[f'{ticker}_price_volume_corr'] = returns.rolling(window).corr(stock_data['Volume'].pct_change())
    
    return features

def preprocess_stock_data(file_path: str):

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    ################################################### 
    #                Partial correlation              #
    ###################################################

    returns_pivot = df.pivot(index='Date', columns='Ticker', values='Close').pct_change().dropna()
 
    
    model = GraphicalLassoCV()
    model.fit(StandardScaler().fit_transform(returns_pivot))
    
    precision = model.precision_
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    
    partial_corr_matrix = pd.DataFrame(
        partial_corr,
        index=returns_pivot.columns,
        columns=returns_pivot.columns
    )
    
    return returns_pivot, partial_corr_matrix


if __name__ == "__main__":
    partial_corr = preprocess_stock_data(STOCK)
    partial_corr.to_csv(OUTPUT/ "PC_matrix_sp500.csv", index=True)
    print("#"*50)
    print(f"Partial correlation matrix has been saved in {OUTPUT} directory.")
    print("#"*50)