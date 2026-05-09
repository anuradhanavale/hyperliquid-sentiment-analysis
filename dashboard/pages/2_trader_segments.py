import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DAILY_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "daily_metrics.parquet"

@st.cache_data
def load_data():
    return pd.read_parquet(DAILY_METRICS_PATH)

st.set_page_config(page_title="Trader Segments", layout="wide")
st.title("👥 Trader Segments – Performance Comparison")

df = load_data()

selected_segment = st.selectbox(
    "Choose segment to analyze",
    ['leverage_segment', 'frequency_segment', 'consistency_segment']
)

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{selected_segment} – Avg Daily PnL")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=df, x=selected_segment, y='daily_pnl', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader(f"{selected_segment} – Win Rate")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x=selected_segment, y='win_rate', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

st.subheader(f"{selected_segment} – Sentiment Breakdown")
seg_sent = df.groupby([selected_segment, 'classification'])['daily_pnl'].mean().reset_index()
fig3, ax3 = plt.subplots()
sns.barplot(data=seg_sent, x=selected_segment, y='daily_pnl', hue='classification', ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

st.subheader("Segment Summary Table")
st.dataframe(df.groupby(selected_segment).agg(
    num_traders=('Account', 'nunique'),
    avg_pnl=('daily_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trades=('trade_count', 'mean'),
    avg_leverage=('avg_leverage', 'mean')
))