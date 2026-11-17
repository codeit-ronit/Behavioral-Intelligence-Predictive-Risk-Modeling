"""Focused checks for time alignment, SQL, saved predictions, and dashboard flows."""
import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.preprocessing import ROOT, OUT, FEATURES, preprocess
from src.modeling import evaluate, labels, temporal_split, thresholds

class PipelineTests(unittest.TestCase):
    def test_cleaning_sql_and_calendar_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            trade=pd.DataFrame({
                'Account':['a','a','a','a','a'], 'Coin':['BTC']*5,
                'Timestamp IST':['01-01-2024 06:00','02-01-2024 06:00','04-01-2024 06:00','bad','01-01-2024 06:00'],
                'Size USD':[100,200,300,50,100], 'Closed PnL':[10,-20,30,4,10],
                'Start Position':[0]*5,'Direction':['Open Long','Close Long','Open Short','Buy','Open Long']})
            trade.to_csv(base/'historical_data.csv',index=False)
            pd.DataFrame({'date':['2023-12-31','2024-01-01','2024-01-03'], 'value':[20,30,40], 'classification':['Fear']*3}).to_csv(base/'fear_greed_index.csv',index=False)
            daily,clean,audit=preprocess(base,base/'out')
            self.assertEqual(audit['exact_duplicates'],1)
            self.assertEqual(audit['invalid_trades_removed'],1)
            self.assertEqual(daily.trade_count.sum(),3)
            self.assertEqual(daily.iloc[0].next_day_abs_pnl,20)
            self.assertTrue(pd.isna(daily.iloc[1].next_day_abs_pnl))
            self.assertEqual(daily.iloc[0].sentiment_value,20)
            self.assertEqual(daily.recent_trades_7d.tolist(),[1,2,3])
            with closing(sqlite3.connect(base/'out'/'analytics.sqlite')) as con:
                volume=con.execute('SELECT SUM(volume) FROM daily_features').fetchone()[0]
            self.assertEqual(volume,600)
            # A future trade and future sentiment cannot alter earlier feature values.
            trade.loc[2,'Closed PnL']=9000
            trade.to_csv(base/'historical_data.csv',index=False)
            sentiment=pd.read_csv(base/'fear_greed_index.csv')
            sentiment.loc[sentiment.date.eq('2024-01-03'),'value']=99
            sentiment.to_csv(base/'fear_greed_index.csv',index=False)
            changed,_,_=preprocess(base,base/'out')
            pd.testing.assert_frame_equal(daily.iloc[:2][FEATURES],changed.iloc[:2][FEATURES])

    def test_utc_rollover_and_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            pd.DataFrame({'Account':['a','a','a'],'Coin':['BTC']*3,
                'Timestamp IST':['02-01-2024 01:00','02-01-2024 06:00','02-01-2024 07:00'],
                'Size USD':[10,None,20],'Closed PnL':[0,5,None], 'Start Position':[0]*3,
                'Direction':['Buy']*3}).to_csv(base/'historical_data.csv',index=False)
            pd.DataFrame({'date':['2023-01-01'],'value':[20],'classification':['Fear']}).to_csv(base/'fear_greed_index.csv',index=False)
            daily,_,audit=preprocess(base,base/'out')
            self.assertEqual(str(daily.iloc[0].date.date()),'2024-01-01')
            self.assertEqual(audit['invalid_trades_removed'],2)
            self.assertTrue(pd.isna(daily.iloc[0].win_rate))
            self.assertTrue(pd.isna(daily.iloc[0].sentiment_value))

    def test_temporal_partitions(self):
        d=pd.read_csv(OUT/'daily_features.csv',parse_dates=['date','target_date'])
        labeled=d.dropna(subset=['next_day_abs_pnl'])
        train,test,cutoff=temporal_split(labeled)
        inner,valid,_=temporal_split(train)
        self.assertLess(train.target_date.max(),test.date.min())
        self.assertLess(inner.target_date.max(),valid.date.min())
        self.assertTrue((pd.to_datetime(d.sentiment_date.dropna()) < d.loc[d.sentiment_date.notna(),'date']).all())
        expected=d[['account','date','daily_pnl']].copy()
        expected['date']-=pd.Timedelta(days=1)
        merged=d.merge(expected,on=['account','date'],suffixes=('','_future'),how='left')
        np.testing.assert_allclose(merged.next_day_abs_pnl,merged.daily_pnl_future.abs(),equal_nan=True)
        meta=json.loads((OUT/'results.json').read_text())
        np.testing.assert_allclose(thresholds(train),meta['thresholds_usd'])

    def test_saved_model_predictions_and_metrics(self):
        data=pd.read_csv(OUT/'daily_features.csv',parse_dates=['date','target_date'])
        _,test,_=temporal_split(data.dropna(subset=['next_day_abs_pnl']).sort_values(['date','account']))
        meta=json.loads((OUT/'results.json').read_text())
        saved=pd.read_csv(OUT/'test_predictions.csv')
        table=pd.read_csv(OUT/'model_comparison.csv').set_index('model')
        y=labels(test,meta['thresholds_usd'])
        for name in table.index:
            model=joblib.load(OUT/(name.lower().replace(' ','_')+'.joblib'))
            prediction=model.predict(test[FEATURES]).astype(int)
            np.testing.assert_array_equal(np.array(meta['classes'])[prediction],saved[name])
            for metric,value in evaluate(y,prediction).items():
                self.assertAlmostEqual(value,table.loc[name,metric],places=12)

    def test_cluster_selection(self):
        scores=pd.read_csv(OUT/'cluster_search.csv')
        meta=json.loads((OUT/'results.json').read_text())['clustering']
        self.assertEqual(int(scores.loc[scores.silhouette.idxmax(),'k']),meta['selected_k'])
        self.assertEqual(pd.read_csv(OUT/'trader_segments.csv').account.nunique(),meta_count())

    def test_dashboard_sections_and_empty_filters(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric),4)
        for section in ['Behavioral Analysis','ML Predictions','Trader Segmentation']:
            app.sidebar.radio[0].set_value(section).run()
            self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('ML Predictions').run()
        for name in ['XGBoost','Majority baseline']:
            app.selectbox[0].set_value(name).run()
            self.assertFalse(app.exception)
        app.selectbox[1].set_value('High').run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Trader Segmentation').run()
        app.multiselect[-1].set_value([]).run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Overview').run()
        app.sidebar.multiselect[0].set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)

def meta_count():
    return json.loads((OUT/'results.json').read_text())['data_audit']['traders']

if __name__=='__main__': unittest.main()
