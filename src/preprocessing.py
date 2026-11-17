"""Clean source CSVs, join lagged sentiment in SQLite, and build daily features."""
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'portfolio'
FEATURES = ['trade_count', 'volume', 'avg_size', 'daily_pnl', 'win_rate',
            'long_short_ratio', 'pnl_std', 'recent_trades_7d', 'sentiment_value']
CLASSES = ['Low', 'Medium', 'High']


def preprocess(data_dir=ROOT / 'data', output_dir=OUT):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_dir / 'historical_data.csv')
    audit = {'raw_trades': len(raw), 'exact_duplicates': int(raw.duplicated().sum())}
    # IDs and epoch timestamps in this export are rounded: do not deduplicate on IDs.
    trades = raw.drop_duplicates().copy()
    timestamp = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    trades['timestamp_utc'] = timestamp.dt.tz_localize('Asia/Kolkata').dt.tz_convert('UTC')
    trades['date'] = trades['timestamp_utc'].dt.strftime('%Y-%m-%d')
    for col in ['Size USD', 'Closed PnL', 'Start Position']:
        trades[col] = pd.to_numeric(trades[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = (trades['Account'].notna() & trades['Account'].str.strip().ne('') &
             trades['date'].notna() & trades['Size USD'].ge(0) & trades['Closed PnL'].notna())
    audit['invalid_trades_removed'] = int((~valid).sum())
    trades = trades.loc[valid].copy()
    # Long/short position direction differs from BUY/SELL (closing shorts is a BUY).
    direction = trades['Direction'].fillna('')
    trades['is_long'] = direction.str.contains('Long', case=False).astype(int)
    trades['is_short'] = direction.str.contains('Short', case=False).astype(int)
    # Reversals contain both words and are excluded from this directional count.
    reversal = trades['is_long'].eq(1) & trades['is_short'].eq(1)
    trades.loc[reversal, ['is_long', 'is_short']] = 0
    sentiment = pd.read_csv(data_dir / 'fear_greed_index.csv')
    sentiment['date'] = pd.to_datetime(sentiment['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    sentiment['value'] = pd.to_numeric(sentiment['value'], errors='coerce')
    sentiment = sentiment.dropna(subset=['date']).drop_duplicates()
    if sentiment['date'].duplicated().any():
        raise ValueError('Conflicting sentiment rows for the same date')
    db_path = output_dir / 'analytics.sqlite'
    with closing(sqlite3.connect(db_path)) as con:
        trades[['Account', 'Coin', 'date', 'Size USD', 'Closed PnL', 'is_long', 'is_short']].to_sql('trades', con, if_exists='replace', index=False)
        sentiment[['date', 'value', 'classification']].to_sql('sentiment', con, if_exists='replace', index=False)
        con.execute('CREATE INDEX IF NOT EXISTS trades_day ON trades(date, Account)')
        query = '''SELECT t.Account AS account, t.date, COUNT(*) AS trade_count,
          SUM(t."Size USD") AS volume, AVG(t."Size USD") AS avg_size,
          SUM(t."Closed PnL") AS daily_pnl,
          SUM(CASE WHEN t."Closed PnL" > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN t."Closed PnL" != 0 THEN 1 ELSE 0 END) AS realized_trades,
          SUM(t.is_long)*1.0/(SUM(t.is_short)+1) AS long_short_ratio,
          AVG(t."Closed PnL"*t."Closed PnL") - AVG(t."Closed PnL")*AVG(t."Closed PnL") AS pnl_variance,
          s.value AS sentiment_value, COALESCE(s.classification, 'Unknown') AS sentiment_category,
          s.date AS sentiment_date
          FROM trades t LEFT JOIN sentiment s ON s.date = date(t.date, '-1 day')
          GROUP BY t.Account, t.date, s.value, s.classification, s.date
          ORDER BY t.Account, t.date'''
        daily = pd.read_sql_query(query, con)
        (output_dir / 'daily_features.sql').write_text(query + ';\n')
        con.execute('CREATE TABLE IF NOT EXISTS daily_features AS ' + query)
        con.execute('DELETE FROM daily_features')
        con.execute('INSERT INTO daily_features ' + query)
        con.commit()
    daily['win_rate'] = daily['wins'].div(daily['realized_trades'].replace(0, np.nan))
    daily['pnl_std'] = np.sqrt(daily.pop('pnl_variance').clip(lower=0))
    daily['date'] = pd.to_datetime(daily['date'])
    recent = []
    for _, group in daily.groupby('account', sort=False):
        recent.extend(group.set_index('date')['trade_count'].rolling('7D').sum().tolist())
    daily['recent_trades_7d'] = recent
    # Self-join on the exact next UTC calendar day; inactive tomorrow is unknown.
    tomorrow = daily[['account', 'date', 'daily_pnl']].rename(columns={'daily_pnl': 'next_day_pnl'})
    tomorrow['date'] = tomorrow['date'] - pd.Timedelta(days=1)
    daily = daily.merge(tomorrow, on=['account', 'date'], how='left', validate='one_to_one')
    daily['target_date'] = daily['date'] + pd.Timedelta(days=1)
    daily['next_day_abs_pnl'] = daily['next_day_pnl'].abs()
    daily['sentiment_value'] = daily['sentiment_value'].where(daily['sentiment_value'].between(0, 100))
    # Keep missing rates/sentiment explicit; model imputation learns on training only.
    daily.to_csv(output_dir / 'daily_features.csv', index=False)
    compact = trades.rename(columns={'Account': 'account', 'Coin': 'coin', 'Size USD': 'volume', 'Closed PnL': 'pnl'})
    compact = compact.merge(sentiment[['date', 'classification']], on='date', how='left', validate='many_to_one')
    compact['classification'] = compact['classification'].fillna('Unknown')
    compact[['account', 'coin', 'date', 'volume', 'pnl', 'is_long', 'is_short', 'classification']].to_parquet(output_dir / 'clean_trades.parquet', index=False)
    audit.update(clean_trades=len(trades), traders=int(trades['Account'].nunique()),
                 daily_rows=len(daily), labeled_rows=int(daily['next_day_abs_pnl'].notna().sum()),
                 missing_lagged_sentiment=int(daily['sentiment_value'].isna().sum()),
                 date_start=str(daily['date'].min().date()), date_end=str(daily['date'].max().date()),
                 total_volume=float(trades['Size USD'].sum()), total_pnl=float(trades['Closed PnL'].sum()))
    return daily, compact, audit
