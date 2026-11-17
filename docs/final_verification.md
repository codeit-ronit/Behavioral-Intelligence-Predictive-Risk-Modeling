# Final bounded verification — 2026-11-17

**Verdict: PASS for all five requested checks. Ready for a technical interview as a scoped portfolio project. No remaining blockers.** No supervised models were retrained, no existing fitted model or metric was changed, and nothing was pushed or deployed.

Machine-readable evidence: [final_verification_evidence.json](final_verification_evidence.json).

| Check | Result | Evidence |
| --- | --- | --- |
| Future-data / target leakage | PASS | Independently reconstructed all nine features from 211,224 source fills across 2,343 trader-days; checked exact calendar targets, sentiment, split boundaries, training-only imputation/thresholds, and saved predictions. |
| Three-seed KMeans stability | PASS | Seeds 7, 42, 2026 each reproduced the saved k=2 partition: sizes 29/3, silhouette 0.8124274574, inertia 69.8810800866; all pairwise adjusted Rand indices and comparisons with saved labels = 1.0. |
| Clean-environment install and dashboard startup | PASS after documented interpreter correction | New Python 3.13.11 venv, no system/user site packages; full requirements.txt installation succeeded; pip check found no broken requirements; core imports succeeded; Streamlit health HTTP 200; Chrome rendered all four sections and the 32-trader scatter without JavaScript errors. |
| README, screenshots, commands, model report | PASS after interpreter correction | Metric/profile tables match saved artifacts; all predictions, probabilities, confusion matrices and per-class metrics replay correctly; feature importance matches fitted models; four PNG screenshots and documentation links checked. |
| Four resume bullets | PASS — 4/4 | Evidence mapping below. |

## Leakage audit and its scope

The documented forecast origin is the end of UTC day t (start of t+1). Same-day realized PnL, frequency, size/volume, win rate, long/short direction counts, and per-fill PnL variability are legitimate features at that origin. They must not be described as start-of-day forecasts. Independent aggregation directly from the timestamped source fills matched every feature row. The trailing activity interval is (t−7 days, t], including today's observed fills, with no future rows.

Targets match the absolute realized PnL sum for the same trader on the exact calendar day t+1. No next-observed-day shift is used. Neither targets, target dates, account identifiers nor full-history clusters enter the model feature list. Full-history KMeans is descriptive only.

Numeric sentiment matches t−1 exactly, with no backfill. All 2,342 matched sentiment timestamps in the supplied file precede the forecast origin. One row has missing lagged sentiment. The source's actual publication/revision history cannot be established from this repository; the code's alignment and source timestamps pass, while vintage provenance remains a data limitation.

The latest outer-training target is **2025-02-02**, before the first test feature date **2025-02-03**. The latest inner-training target is **2024-11-24**, before the first validation feature date **2024-11-25**. Saved imputer statistics match medians of outer training only. Training-derived thresholds reproduce **USD 101.859038** and **USD 1196.800511**. Code inspection confirms inner tuning learns inner thresholds and imputation only from inner training, selects among three configurations per model, and does not use holdout scores for tuning.

The 832 saved holdout predictions, class labels, probabilities, confusion matrices and per-class reports reproduced without fitting. Recomputed aggregate metrics match saved values within 1e-12:

| Model | Accuracy | Macro-F1 | Macro precision | Macro recall |
| --- | --- | --- | --- | --- |
| Majority baseline | 0.3858173077 | 0.1856027754 | 0.1286057692 | 0.3333333333 |
| Random Forest | 0.4807692308 | 0.4809749858 | 0.4913999044 | 0.4847362060 |
| XGBoost | 0.4591346154 | 0.4590284340 | 0.4792470161 | 0.4669475136 |

Dropping unknown next-day outcomes conditions evaluation on future observed activity. This is disclosed **selection bias**, rather than a future-valued predictor. Results do not establish performance for inactive days, unseen accounts, or a full exchange population.

## Stability scope

Only three small KMeans fits were performed at the existing selected k=2 using the existing feature transformation/scaler. No saved clustering artifacts were replaced. This establishes initialization stability on these 32 accounts, not resampling or future-population stability. Total PnL and signed-log scaling strongly separate positive and negative outcomes; high silhouette does not establish trader skill or distinct professional types. The three-account segment warrants caution.

## Clean installation and exact resolved blocker

The literal documented `python3.13 -m venv ...` initially failed with **`zsh:1: command not found: python3.13`**. The installed Conda Python 3.13 interpreter was used by its absolute executable path to create an isolated environment. The README and generator explain how to locate an installed interpreter without assuming a machine-specific directory. In the recorded commands below, `VERIFY_ENV` denotes the temporary isolated environment; machine-specific paths have been normalized for public documentation:

```bash
"$VERIFY_ENV/bin/python" -m pip install -r requirements.txt
"$VERIFY_ENV/bin/python" -m pip check
"$VERIFY_ENV/bin/python" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8517 --server.headless=true --browser.gatherUsageStats=false
```

All passed. The port and flags isolate the test server; the application entrypoint is the documented one. The temporary test server was stopped. Core library versions match the pinned requirements and existing artifact versions. This was a fresh Python dependency environment using **existing local outputs**, not a new checkout with another full training run. The documented initial modeling command was deliberately not repeated; source preprocessing/model code and saved artifacts were already available and passed the targeted audit.

The first browser smoke attempt asserted chart counts before rendering completed. Waiting for the expected chart count resolved that test synchronization issue; only the smoke check was repeated. No application change was needed.

## Documentation and screenshot checks

README and model-report metric/profile tables match CSV artifacts; confusion matrices, per-class reports, feature importance, thresholds and split descriptions match implementation and saved outputs. Source CSV fingerprints also match results.json. Four existing 1440×1050 PNG screenshots show the actual section headers and artifact values (including 211,224 trades, 32 traders, 83.2% nonzero-fill win rate, the model comparison and 0.812 silhouette). They are viewport captures, not complete captures of content below the visible area. Current clean-environment browser rendering confirmed the displayed totals, evaluation row counts and trader scatter. Screenshots were not replaced. Local documentation links and Python syntax checks passed.

Only documentation was corrected: the interpreter fallback, and wording distinguishing unimplemented **resampling stability** from the three-seed initialization check now completed. The original implementation verification log remains historical and links here for current status.

## Resume evidence — four verified bullets

| Proposed bullet | Result | Implementation and evaluation evidence |
| --- | --- | --- |
| Built a reproducible Pandas and SQLite pipeline processing 211,224 trading records into behavioral features with historical time alignment. | PASS | src/preprocessing.py, daily_features.sql, source fingerprints; independent nine-feature reconstruction and split/sentiment audit. |
| Compared Random Forest and XGBoost using chronological validation, bounded tuning, and a majority baseline; achieved 0.481 holdout macro-F1 with Random Forest. | PASS | src/modeling.py, search_results.csv, model_comparison.csv; saved prediction replay gives macro-F1 0.480974985816047. |
| Applied transformed, scaled KMeans clustering and silhouette analysis to identify two descriptive trader segments. | PASS | src/modeling.py, kmeans.joblib, cluster_search.csv, trader_segments.csv and cluster_profiles.csv; three-seed initialization agreement = 1.0 ARI. |
| Developed a four-section Streamlit and Plotly dashboard displaying saved predictions, model metrics, feature importance, and behavioral insights. | PASS | app.py, existing screenshots and clean-environment Chrome smoke across all four sections; artifacts loaded without training. |

For the interview, explain the after-day-end forecast origin, the absolute-realized-PnL target (including large profits), conditional activity selection, the 32-account sample, moderate model performance, and the descriptive nature of full-history clustering. Do not claim fraud detection, market-price volatility forecasting, calibrated probabilities, or production readiness.

## Presentation review addendum — 2026-11-17

The model/data verification above remains unchanged. A subsequent presentation pass moved verified results, technologies, setup navigation and a representative screenshot to the README opening; retained detailed methodology in expandable sections; and updated the README generator to preserve the layout. The daily PnL chart now uses a visible line instead of densely packed bars, and short captions explain macro-F1, silhouette and the small sample. No model, data, target, split or saved evaluation values changed.

One targeted browser smoke rendered all four sections at 1440×1050 and 1280×800 with no JavaScript errors or horizontal overflow. Four desktop viewport screenshots were refreshed and the rendered README opening/results were visually reviewed. The evidence JSON retains earlier screenshot fingerprints and records the refreshed fingerprints and presentation review separately. All four resume bullets remain supported. Only changed-file syntax, generated metric formatting, Markdown links/anchors and Git hygiene were checked; the leakage, clustering and clean-install audits were not repeated.

No tracked environments or high-confidence credential matches were found; no new files over 10 MiB were introduced. Existing large source/legacy CSV exports were preserved. The intended repository's main branch was read without changing remote content: it contains two README commits absent from the local branch (remote tip `c453998`). Local work can be committed, but publishing requires synchronization first; no merge, history rewrite or push was performed.
