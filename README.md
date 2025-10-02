## 📊 Clustering & Cointegration Analysis of Indian Equities (2016–2024)

This project analyzes systemic risk clusters and pair trading opportunities in the Indian equity market using hierarchical clustering and econometric validation.

🔹 Methodology

Partial Correlation Matrices → capture conditional dependencies among stocks.

Hierarchical Clustering → applied on full NIFTY50 and sectoral subsets (IT, Banking).

Validation Pipeline:

Bootstrap Stability (0.55 in IT sector)

Monte Carlo Null Model (Significant for IT sector: p=0.011, modularity ~0.45)

Temporal Validation (Out-of-sample ARI ~0.22, reflecting regime shifts 2018–2024).

Cointegration Analysis: intra-cluster stock pairs tested for long-term mean reversion.

## 📈 Key Results

IT Sector → clear, statistically significant clustering structure.

Banking Sector & Full NIFTY50 → clustering not statistically significant in the same time frame, suggesting higher structural heterogeneity.

Temporal ARI ~0.22 → clusters evolve across market regimes (COVID, liquidity shocks).

Cointegration pipeline highlights potential tradable pairs but requires multiple-testing adjustments.

## Future Work

Extend to rolling-window analysis to track how sectoral structures evolve in crises.

Sharpe ratio backtesting of cointegrated pairs.

Cross-sector comparison of clustering robustness.