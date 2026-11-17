"""Streamlit exploration of saved analytics; never trains models on refresh."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Behavioral Analytics & Predictive Risk Modeling', page_icon='📊', layout='wide')
OUT = Path(__file__).resolve().parent / 'outputs' / 'portfolio'
COLORS = ['#38bdf8', '#2dd4bf', '#fbbf24', '#a78bfa', '#fb7185']
px.defaults.template = 'plotly_white'
px.defaults.color_discrete_sequence = COLORS
st.markdown('''<style>.block-container {padding-top:2rem;max-width:1450px}
[data-testid="stMetric"] {background:#f0f7ff;border:1px solid #d5e5f4;border-radius:12px;padding:18px}
h1 {letter-spacing:-.04em} </style>''', unsafe_allow_html=True)

@st.cache_data
def load_data(version):
    return (pd.read_parquet(OUT/'clean_trades.parquet'), pd.read_csv(OUT/'trader_segments.csv'),
            pd.read_csv(OUT/'model_comparison.csv'), pd.read_csv(OUT/'test_predictions.csv'),
            pd.read_csv(OUT/'feature_importance.csv'), json.loads((OUT/'results.json').read_text()))

if not (OUT/'results.json').exists():
    st.info('Generate the saved analysis first: python -m src.modeling')
    st.stop()
trades, segments, metrics, predictions, importance, results = load_data((OUT/'results.json').stat().st_mtime_ns)
trades['date'] = pd.to_datetime(trades['date'])
st.sidebar.title('Explore the dataset')
section = st.sidebar.radio('Section', ['Overview', 'Behavioral Analysis', 'ML Predictions', 'Trader Segmentation'])
dates = st.sidebar.date_input('Trade dates (UTC)', value=(trades.date.min().date(), trades.date.max().date()), min_value=trades.date.min().date(), max_value=trades.date.max().date())
sentiments = st.sidebar.multiselect('Same-day sentiment', sorted(trades.classification.unique()), default=sorted(trades.classification.unique()))
accounts = st.sidebar.multiselect('Traders (empty = all)', sorted(trades.account.unique()), format_func=lambda s:s[:10]+'…')
filtered = trades[trades.classification.isin(sentiments)]
if len(dates) == 2:
    filtered = filtered[filtered.date.between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))]
if accounts:
    filtered = filtered[filtered.account.isin(accounts)]
st.sidebar.caption('Trade filters apply to Overview and Behavioral Analysis. Model evaluation and segments use their saved analysis populations.')
st.title('Behavioral Analytics & Predictive Risk Modeling')
st.caption('HYPERLIQUID · TRADER BEHAVIOR · HISTORICAL RISK · REPRODUCIBLE ML')
st.divider()

if section in ['Overview', 'Behavioral Analysis'] and filtered.empty:
    st.info('No trades match these filters. Choose a wider date range or another sentiment.')
    st.stop()

def chart(fig):
    fig.update_layout(margin=dict(l=15,r=15,t=40,b=15), paper_bgcolor='rgba(0,0,0,0)', legend_title_text='')
    st.plotly_chart(fig, width='stretch', theme=None)

if section == 'Overview':
    st.subheader('A clear view of observed trading behavior')
    cols = st.columns(4)
    for col, label, val in zip(cols, ['Trades','Traders','Trading volume','Realized PnL'], [f'{len(filtered):,}', f'{filtered.account.nunique():,}', f'${filtered.volume.sum()/1e9:,.3f}B', f'${filtered.pnl.sum()/1e6:,.2f}M']):
        col.metric(label,val)
    st.caption(f"Exact totals: volume USD {filtered.volume.sum():,.2f} · realized PnL USD {filtered.pnl.sum():,.2f}")
    left,right=st.columns([1.5,1])
    daily = filtered.groupby('date',as_index=False).agg(volume=('volume','sum'),trades=('pnl','size'))
    with left:
        chart(px.area(daily,x='date',y='volume',title='Daily notional trading volume',labels={'volume':'USD','date':'UTC date'}))
        st.caption('Notional turnover measures observed activity; it is not account equity or protocol revenue.')
    with right:
        counts=filtered.classification.value_counts().rename_axis('sentiment').reset_index(name='trades')
        chart(px.pie(counts,names='sentiment',values='trades',hole=.6,title='Sentiment distribution by trade'))
        st.caption('Same-day sentiment is descriptive. The predictive models use lagged sentiment.')
    st.info(f"Source coverage: {results['data_audit']['date_start']} to {results['data_audit']['date_end']} · Exact duplicate rows removed: {results['data_audit']['exact_duplicates']:,}.")

elif section == 'Behavioral Analysis':
    st.subheader('Activity, profitability, and market sentiment')
    realized=filtered[filtered.pnl.ne(0)]
    st.metric('Win rate on nonzero realized-PnL fills',f'{realized.pnl.gt(0).mean():.1%}' if len(realized) else 'N/A')
    daily=filtered.groupby('date',as_index=False).agg(trades=('pnl','size'),pnl=('pnl','sum'))
    left,right=st.columns(2)
    with left: chart(px.line(daily,x='date',y='trades',title='Trading frequency over time',labels={'date':'UTC date','trades':'Fills'}))
    with right: chart(px.line(daily,x='date',y='pnl',title='Daily realized PnL (before fees)',color_discrete_sequence=['#0f766e'],labels={'date':'UTC date','pnl':'Realized PnL (USD)'}))
    grouped=filtered.groupby('classification',as_index=False).agg(volume=('volume','sum'),pnl=('pnl','sum'),trades=('pnl','size'),days=('date','nunique'))
    grouped['pnl_per_1000']=1000*grouped.pnl/grouped.volume.replace(0,float('nan'))
    grouped['avg_daily_volume']=grouped.volume/grouped.days
    left,right=st.columns(2)
    with left: chart(px.bar(grouped,x='classification',y='pnl_per_1000',color='classification',title='Realized PnL per $1,000 traded'))
    with right: chart(px.bar(grouped,x='classification',y='avg_daily_volume',color='classification',title='Average daily volume by sentiment'))
    st.caption('Associations in this selected sample do not establish causation or a profitable trading strategy. Zero-PnL fills are excluded from win-rate calculations; fees are not deducted.')
    st.dataframe(grouped,width='stretch',hide_index=True)

elif section == 'ML Predictions':
    st.subheader('Predicting next-day absolute realized PnL')
    low,high=results['thresholds_usd']
    st.write(f'At the end of UTC day t, classify |realized PnL| on calendar day t+1: Low ≤ USD {low:,.2f}; Medium ≤ USD {high:,.2f}; High above USD {high:,.2f}. Only traders observed on both days have known outcomes.')
    st.caption(f"Chronological holdout: {results['test_start']}–{results['test_end']} · {results['train_rows']:,} training rows / {results['test_rows']:,} test rows. Thresholds, imputation, and tuning use training history. Labels crossing split boundaries are purged.")
    st.dataframe(metrics.style.format({c:'{:.3f}' for c in metrics.columns if c!='model'}),width='stretch',hide_index=True)
    st.caption(f"Macro-F1 gives Low, Medium and High equal weight; it can hide weak recall for one class and does not assess probability calibration. This selected {results['data_audit']['traders']}-trader sample evaluates future activity, with accounts potentially shared across training and test.")
    model=st.selectbox('Model',['Random Forest','XGBoost','Majority baseline'])
    left,right=st.columns(2)
    with left:
        cm=results['reports'][model]['confusion_matrix']
        chart(px.imshow(cm,x=results['classes'],y=results['classes'],text_auto=True,color_continuous_scale='Teal',title='Confusion matrix',labels={'x':'Predicted','y':'Actual','color':'Rows'}))
    with right:
        if model!='Majority baseline':
            chart(px.bar(importance[importance.model.eq(model)].sort_values('importance'),x='importance',y='feature',orientation='h',title='Model feature importance'))
        else: st.info('The baseline always predicts the most frequent training class.')
    st.caption('Feature importance is model-specific and does not measure causation. Absolute PnL reflects size and activity as well as risk; large profits also receive High labels.')
    prediction_filter=st.selectbox('Prediction class',['All','Low','Medium','High'])
    shown=predictions if prediction_filter=='All' else predictions[predictions[model].eq(prediction_filter)]
    st.dataframe(shown[['account','date','target_date','actual',model,model+' high_probability']],width='stretch',hide_index=True)
    st.download_button('Download holdout predictions',predictions.to_csv(index=False),'test_predictions.csv','text/csv')
    st.caption('Saved holdout predictions are retrospective evaluation, not a live forecast service. Probabilities are uncalibrated.')

else:
    st.subheader('Trader segments discovered from behavior')
    clustering=results['clustering']
    st.write(f"KMeans selected {clustering['selected_k']} clusters from k=2…6 using silhouette score ({clustering['silhouette']:.3f}).")
    st.caption(f"Silhouette compares similarity within groups with separation between groups; higher suggests clearer separation under these features. These {results['data_audit']['traders']} traders form a small descriptive sample.")
    st.caption('Features: activity, average trade size, total realized PnL, per-fill PnL standard deviation, and win rate. Signed log transforms reduce skew; RobustScaler handles scale differences. Segments describe the full dataset and do not enter historical risk predictions.')
    options=sorted(segments.segment.unique())
    chosen=st.multiselect('Segments',options,default=options)
    subset=segments[segments.segment.isin(chosen)]
    if subset.empty:
        st.info('Select at least one segment.')
    else:
        chart(px.scatter(subset,x='Activity',y='Total_PnL',size='Avg_Size',color='segment',hover_name='account',log_x=True,size_max=45,title='Activity versus realized profitability'))
        for c,description in clustering['descriptions'].items(): st.write(f'**Segment {c}:** {description}')
        st.dataframe(pd.read_csv(OUT/'cluster_profiles.csv'),width='stretch',hide_index=True)
        chart(px.line(pd.read_csv(OUT/'cluster_search.csv'),x='k',y='silhouette',markers=True,title='Cluster-count selection'))
        st.caption('Segment numbers are arbitrary. These patterns do not verify professional status, trading skill, fraud, or institutional ownership.')
