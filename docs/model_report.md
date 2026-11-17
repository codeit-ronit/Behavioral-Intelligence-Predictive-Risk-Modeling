# Model comparison and behavioral findings

Generated from saved artifacts by `python -m src.reporting`. See the root README for setup and data provenance.

## Procedure

Predictions are made after UTC day t closes. The target is the absolute sum of realized PnL on **calendar day t+1**, conditional on that trader having recorded fills on both days. Inactive next days remain unknown, not zero. The final day's outcome is unknown.

Training tertiles define Low ≤ **$101.86**, Medium ≤ **$1,196.80**, and High above that. These are absolute monetary exposure buckets, not return volatility or downside-only risk.

Split on the first 80% of unique eligible feature dates, reserving the remaining dates for the test set. Unequal daily activity means this is not an 80/20 row split. Training: **2023-12-13–2025-02-01 (862 rows)**. Holdout feature dates: **2025-02-03–2025-04-30 (832 rows)**. Require every training outcome date to precede the next partition's first feature date. 10 boundary rows are purged.

Within training, another chronological date split starts validation on **2024-11-25** (398 inner-training / 462 validation rows). Fit inner thresholds and median imputation on inner training only. Compare three configurations per model using validation macro-F1. Refit the chosen configuration on all outer training data, relearning thresholds and imputation there. Evaluate the reserved holdout once. Both models use 120 trees, seed 42 and at most two threads. Random Forest uses balanced class weights; XGBoost uses its default unweighted multiclass objective. This compares practical configurations, not a controlled algorithm benchmark.

Selected parameters: `{"Random Forest": {"max_depth": 5, "min_samples_leaf": 3}, "XGBoost": {"learning_rate": 0.05, "max_depth": 2}}`. A `DummyClassifier(strategy="most_frequent")` supplies the majority-class baseline. Macro metrics weight all three classes equally; per-class precision and recall and confusion matrices are saved in `results.json`.

## Holdout comparison

| model | accuracy | macro_f1 | macro_precision | macro_recall |
| --- | --- | --- | --- | --- |
| Majority baseline | 0.3858 | 0.1856 | 0.1286 | 0.3333 |
| Random Forest | 0.4808 | 0.4810 | 0.4914 | 0.4847 |
| XGBoost | 0.4591 | 0.4590 | 0.4792 | 0.4669 |

### Majority baseline

Rows = actual; columns = predicted; class order = Low, Medium, High.

| Actual | Predicted Low | Predicted Medium | Predicted High |
| --- | --- | --- | --- |
| Low | 0 | 0 | 265 |
| Medium | 0 | 0 | 246 |
| High | 0 | 0 | 321 |

| Class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Low | 0.0000 | 0.0000 | 0.0000 | 265.0000 |
| Medium | 0.0000 | 0.0000 | 0.0000 | 246.0000 |
| High | 0.3858 | 1.0000 | 0.5568 | 321.0000 |

### Random Forest

Rows = actual; columns = predicted; class order = Low, Medium, High.

| Actual | Predicted Low | Predicted Medium | Predicted High |
| --- | --- | --- | --- |
| Low | 130 | 85 | 50 |
| Medium | 74 | 129 | 43 |
| High | 72 | 108 | 141 |

| Class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Low | 0.4710 | 0.4906 | 0.4806 | 265.0000 |
| Medium | 0.4006 | 0.5244 | 0.4542 | 246.0000 |
| High | 0.6026 | 0.4393 | 0.5081 | 321.0000 |

### XGBoost

Rows = actual; columns = predicted; class order = Low, Medium, High.

| Actual | Predicted Low | Predicted Medium | Predicted High |
| --- | --- | --- | --- |
| Low | 137 | 89 | 39 |
| Medium | 83 | 127 | 36 |
| High | 99 | 104 | 118 |

| Class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Low | 0.4295 | 0.5170 | 0.4692 | 265.0000 |
| Medium | 0.3969 | 0.5163 | 0.4488 | 246.0000 |
| High | 0.6114 | 0.3676 | 0.4591 | 321.0000 |

## Feature importance

| model | feature | importance |
| --- | --- | --- |
| Random Forest | pnl_std | 0.1955 |
| Random Forest | recent_trades_7d | 0.1665 |
| Random Forest | daily_pnl | 0.1592 |
| Random Forest | trade_count | 0.1226 |
| Random Forest | avg_size | 0.0883 |
| Random Forest | volume | 0.0851 |
| Random Forest | sentiment_value | 0.0847 |
| Random Forest | long_short_ratio | 0.0712 |
| Random Forest | win_rate | 0.0270 |
| XGBoost | pnl_std | 0.1705 |
| XGBoost | daily_pnl | 0.1662 |
| XGBoost | recent_trades_7d | 0.1568 |
| XGBoost | sentiment_value | 0.1084 |
| XGBoost | trade_count | 0.0842 |
| XGBoost | win_rate | 0.0831 |
| XGBoost | volume | 0.0806 |
| XGBoost | long_short_ratio | 0.0772 |
| XGBoost | avg_size | 0.0730 |

Impurity/gain importance can favor continuous variables and correlated features; values are not comparable measures of effect across algorithms.

## KMeans selection

| k | silhouette |
| --- | --- |
| 2 | 0.8124 |
| 3 | 0.3038 |
| 4 | 0.3135 |
| 5 | 0.3367 |
| 6 | 0.3328 |

| Cluster | Activity | Avg_Size | Total_PnL | PnL_Volatility | Win_Rate | Traders |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 6965.4828 | 6187.1073 | 364352.4089 | 849.8294 | 0.8650 | 29 |
| 1 | 3075.0000 | 4275.8758 | -89753.6387 | 916.2468 | 0.7062 | 3 |

- Segment 0: Higher activity, larger trades; positive mean realized PnL; 87% mean win rate.
- Segment 1: Lower activity, larger trades; non-positive mean realized PnL; 71% mean win rate.

## Sentiment associations

| classification | volume | pnl | days | trades | avg_daily_volume | pnl_per_1000 |
| --- | --- | --- | --- | --- | --- | --- |
| Extreme Fear | 116426382.5800 | 1072333.9772 | 14 | 21303 | 8316170.1843 | 9.2104 |
| Extreme Greed | 127164843.3700 | 2615121.0907 | 112 | 40180 | 1135400.3872 | 20.5648 |
| Fear | 486348556.7800 | 2868016.1023 | 89 | 61510 | 5464590.5256 | 5.8970 |
| Greed | 269505939.1600 | 2439462.9915 | 197 | 48668 | 1368050.4526 | 9.0516 |
| Neutral | 191741720.5700 | 1302024.7817 | 64 | 39563 | 2995964.3839 | 6.7905 |

## Findings

- Observed notional turnover totals **$1,191,187,442.46** across **211,224 fills** and **32 traders**. Realized PnL totals **$10,296,958.94**, before fees.
- **Extreme Fear** has the largest average daily volume among observed sentiment groups (**$8,316,170**). This is an association in this sample, not evidence of causality.
- The fill-weighted win rate among nonzero realized-PnL fills is **83.2%**. Zero-PnL fills are excluded; this is not a position-level success rate.
- Random Forest exceeds the majority baseline by **9.50 percentage points** in holdout accuracy. Its three largest impurity-based feature importances are **pnl_std, recent_trades_7d, daily_pnl**.
- KMeans selects **2 segments** with silhouette **0.812**. Segment sizes are **29, 3**; the smaller group limits interpretation.

## Limitations

- Only 32 selected traders: results may not generalize to the wider platform, new accounts, or future market regimes. The same accounts may occur in training and test; this evaluates future behavior, not unseen-trader generalization.
- Realized PnL depends on trade size, inventory closure and execution behavior. Large profits also count as High; there are no equity, leverage, complete position histories, verified fraud labels, or reliable mark-to-market returns. This is not fraud detection.
- Conditioning on observed activity tomorrow excludes inactive days and can introduce selection bias. Class boundaries are sample-dependent. High-class recall is modest; predictions are exploratory and probabilities are uncalibrated.
- The export rounds epoch timestamps and trade IDs. Explicit IST timestamps are used instead; only exact row duplicates are removed. Minute-level timing and an unverified export history limit precision.
- Realized PnL is before fees. Win rate is based on nonzero-PnL fills, not completed positions. Sentiment is a broad Bitcoin proxy and may not represent every asset. One daily row lacks lagged sentiment and is imputed only from training data.
- Clustering uses the full dataset for descriptive exploration, including total PnL. Signed-log scaling separates positive and negative monetary outcomes strongly; high silhouette alone does not establish stable behavioral types. There are only three accounts in the smaller cluster.
- The original notebook and its historical exports remain for provenance. Its old random-split scores, assumed archetype names, and untested strategy recommendations are not validated claims of this version.

Future improvements: gather a broader trader sample and account equity, evaluate rolling temporal holdouts and unseen accounts, test cluster stability under resampling, and calibrate probabilities using training history. These are not implemented.
