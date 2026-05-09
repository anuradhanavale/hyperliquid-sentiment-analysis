import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DAILY_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "daily_metrics.parquet"

st.set_page_config(layout="wide")
st.title("Hyperliquid Sentiment & Trader Behavior Dashboard")

@st.cache_data
def load_data():
    df = pd.read_parquet(DAILY_METRICS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

st.sidebar.header("Filters")
selected_sentiment = st.sidebar.multiselect(
    "Market Sentiment",
    options=df['classification'].unique(),
    default=df['classification'].unique()
)
selected_segments = st.sidebar.multiselect(
    "Leverage Segment",
    options=df['leverage_segment'].dropna().unique(),
    default=df['leverage_segment'].dropna().unique()
)

filtered = df[df['classification'].isin(selected_sentiment) & df['leverage_segment'].isin(selected_segments)]

col1, col2, col3 = st.columns(3)
col1.metric("Avg Daily PnL (Fear)", f"${filtered[filtered['classification']=='Fear']['daily_pnl'].mean():.2f}")
col2.metric("Avg Daily PnL (Greed)", f"${filtered[filtered['classification']=='Greed']['daily_pnl'].mean():.2f}")
col3.metric("Overall Win Rate", f"{filtered['win_rate'].mean():.2%}")

st.subheader("Daily PnL Over Time")
fig1, ax1 = plt.subplots(figsize=(12,5))
sns.lineplot(data=filtered, x='date', y='daily_pnl', hue='classification', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Trader Behavior by Sentiment")
col_left, col_mid, col_right = st.columns(3)
with col_left:
    fig2, ax2 = plt.subplots()
    sns.barplot(data=filtered, x='classification', y='avg_leverage', ax=ax2)
    ax2.set_title("Leverage")
    st.pyplot(fig2)
with col_mid:
    fig3, ax3 = plt.subplots()
    sns.barplot(data=filtered, x='classification', y='trade_count', ax=ax3)
    ax3.set_title("Trade Count")
    st.pyplot(fig3)
with col_right:
    fig4, ax4 = plt.subplots()
    sns.barplot(data=filtered, x='classification', y='long_short_ratio', ax=ax4)
    ax4.set_title("Long/Short Ratio")
    st.pyplot(fig4)

st.subheader("Performance by Leverage Segment")
seg_pnl = filtered.groupby(['classification', 'leverage_segment'])['daily_pnl'].mean().reset_index()
fig5, ax5 = plt.subplots()
sns.barplot(data=seg_pnl, x='classification', y='daily_pnl', hue='leverage_segment', ax=ax5)
st.pyplot(fig5)

if st.checkbox("Show raw daily metrics"):
    st.dataframe(filtered)