"""Bounded temporal model selection and descriptive trader clustering."""
import json
import hashlib
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, silhouette_score
from xgboost import XGBClassifier
import xgboost
from .preprocessing import FEATURES, CLASSES, OUT, ROOT, preprocess

SEED = 42


def temporal_split(data, fraction=.8):
    dates = sorted(data['date'].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * fraction)])
    # Purge outcomes not observed strictly before the next partition starts.
    return data[data['target_date'] < cutoff].copy(), data[data['date'] >= cutoff].copy(), cutoff


def thresholds(data):
    cuts = data['next_day_abs_pnl'].quantile([1/3, 2/3]).to_numpy()
    if cuts[0] >= cuts[1]:
        raise ValueError('Training outcomes do not support three distinct risk buckets')
    return cuts


def labels(data, cuts):
    return np.searchsorted(cuts, data['next_day_abs_pnl'].to_numpy(), side='left')


def evaluate(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, labels=[0,1,2], average='macro', zero_division=0),
            'macro_precision': precision_score(y, pred, labels=[0,1,2], average='macro', zero_division=0),
            'macro_recall': recall_score(y, pred, labels=[0,1,2], average='macro', zero_division=0)}


def train_models(daily, output_dir=OUT):
    data = daily.dropna(subset=['next_day_abs_pnl']).sort_values(['date', 'account'])
    train, test, cutoff = temporal_split(data)
    inner, validation, val_cutoff = temporal_split(train)
    inner_cuts, cuts = thresholds(inner), thresholds(train)
    search, selected = [], {}
    candidates = {
        'Random Forest': [dict(max_depth=d, min_samples_leaf=leaf) for d, leaf in [(5, 3), (8, 5), (None, 8)]],
        'XGBoost': [dict(max_depth=d, learning_rate=lr) for d, lr in [(2,.05), (3,.05), (3,.1)]]}
    def build(name, params):
        if name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=120, class_weight='balanced', random_state=SEED, n_jobs=2, **params)
        else:
            model = XGBClassifier(n_estimators=120, tree_method='hist', objective='multi:softprob', num_class=3,
                                  eval_metric='mlogloss', random_state=SEED, n_jobs=2, **params)
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), model)
    for name, configs in candidates.items():
        scores = []
        for params in configs:
            model = build(name, params)
            model.fit(inner[FEATURES], labels(inner, inner_cuts))
            score = evaluate(labels(validation, inner_cuts), model.predict(validation[FEATURES]))['macro_f1']
            search.append({'model': name, 'parameters': json.dumps(params), 'validation_macro_f1': score})
            scores.append(score)
        selected[name] = configs[int(np.argmax(scores))]
    pd.DataFrame(search).to_csv(output_dir / 'search_results.csv', index=False)
    models = {'Majority baseline': make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), DummyClassifier(strategy='most_frequent'))}
    models.update({name: build(name, params) for name, params in selected.items()})
    rows, importance, reports = [], [], {}
    predictions = test[['account', 'date', 'target_date', 'next_day_abs_pnl']].copy()
    predictions['actual'] = np.array(CLASSES)[labels(test, cuts)]
    for name, model in models.items():
        model.fit(train[FEATURES], labels(train, cuts))
        pred = model.predict(test[FEATURES]).astype(int)
        rows.append({'model': name, **evaluate(labels(test, cuts), pred)})
        predictions[name] = np.array(CLASSES)[pred]
        predictions[name + ' high_probability'] = model.predict_proba(test[FEATURES])[:,2]
        reports[name] = {'confusion_matrix': confusion_matrix(labels(test, cuts), pred, labels=[0,1,2]).tolist(),
                         'per_class': classification_report(labels(test, cuts), pred, labels=[0,1,2], target_names=CLASSES, output_dict=True, zero_division=0)}
        joblib.dump(model, output_dir / (name.lower().replace(' ', '_') + '.joblib'))
        if name != 'Majority baseline':
            importance.extend({'model': name, 'feature': f, 'importance': float(v)} for f,v in zip(FEATURES, model[-1].feature_importances_))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / 'model_comparison.csv', index=False)
    predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    pd.DataFrame(importance).to_csv(output_dir / 'feature_importance.csv', index=False)
    metadata = {'seed': SEED, 'features': FEATURES, 'classes': CLASSES, 'thresholds_usd': cuts.tolist(),
                'train_rows': len(train), 'test_rows': len(test), 'inner_train_rows':len(inner), 'validation_rows':len(validation),
                'test_start': str(cutoff.date()), 'validation_start':str(val_cutoff.date()),
                'train_start':str(train['date'].min().date()), 'train_end':str(train['date'].max().date()),
                'test_end':str(test['date'].max().date()), 'selected_parameters': selected,
                'test_class_counts': predictions['actual'].value_counts().to_dict(), 'reports': reports}
    return metadata


def segment(trades, output_dir=OUT):
    grouped = trades.groupby('account')
    dna = grouped.agg(Activity=('pnl','size'), Avg_Size=('volume','mean'), Total_PnL=('pnl','sum'), PnL_Volatility=('pnl','std'))
    realized = trades[trades['pnl'] != 0].groupby('account')['pnl'].agg(lambda s: s.gt(0).mean())
    dna['Win_Rate'] = realized.reindex(dna.index).fillna(0)
    dna = dna.fillna(0)
    # Signed log compresses extreme monetary scales before robust scaling.
    transformed = np.sign(dna) * np.log1p(dna.abs())
    transformed['Win_Rate'] = dna['Win_Rate']
    scaler = RobustScaler()
    scaled = scaler.fit_transform(transformed)
    scores, fitted = [], {}
    for k in range(2, min(6, len(dna)-1)+1):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        pred = model.fit_predict(scaled)
        scores.append({'k': k, 'silhouette': silhouette_score(scaled, pred)})
        fitted[k] = model
    best = max(scores, key=lambda r:r['silhouette'])
    dna['Cluster'] = fitted[best['k']].labels_
    profile = dna.groupby('Cluster').mean()
    profile['Traders'] = dna.groupby('Cluster').size()
    descriptions = {}
    for c, row in profile.iterrows():
        activity = 'Higher activity' if row.Activity >= dna.Activity.median() else 'Lower activity'
        size = 'larger trades' if row.Avg_Size >= dna.Avg_Size.median() else 'smaller trades'
        pnl = 'positive' if row.Total_PnL > 0 else 'non-positive'
        descriptions[int(c)] = f'{activity}, {size}; {pnl} mean realized PnL; {row.Win_Rate:.0%} mean win rate.'
    dna['segment'] = dna['Cluster'].map(lambda c:f'Segment {c}')
    dna.reset_index().to_csv(output_dir / 'trader_segments.csv', index=False)
    profile.to_csv(output_dir / 'cluster_profiles.csv')
    pd.DataFrame(scores).to_csv(output_dir / 'cluster_search.csv', index=False)
    joblib.dump({'scaler':scaler, 'kmeans':fitted[best['k']], 'features':list(transformed.columns)}, output_dir / 'kmeans.joblib')
    return {'selected_k': best['k'], 'silhouette': best['silhouette'], 'descriptions': descriptions}


def main():
    daily, trades, audit = preprocess()
    metadata = train_models(daily)
    metadata['clustering'] = segment(trades)
    metadata['data_audit'] = audit
    metadata['versions'] = {'python': platform.python_version(), 'pandas':pd.__version__, 'numpy':np.__version__, 'scikit-learn':sklearn.__version__, 'xgboost':xgboost.__version__}
    metadata['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')}
    (OUT/'results.json').write_text(json.dumps(metadata, indent=2))
    from .reporting import main as write_report
    write_report()
    print(pd.read_csv(OUT/'model_comparison.csv').to_string(index=False))
    print(json.dumps({'data':audit, 'clustering':metadata['clustering']}, indent=2))

if __name__ == '__main__':
    main()
