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

st.set_page_config(page_title="Overview", layout="wide")
st.title("📈 Sentiment & Trader Behavior – Overview")

df = load_data()
df['date'] = pd.to_datetime(df['date'])

col1, col2, col3 = st.columns(3)
col1.metric("Total Traders", df['Account'].nunique())
col2.metric("Total Trading Days", df['date'].nunique())
col3.metric("Avg Daily PnL (All)", f"${df['daily_pnl'].mean():.2f}")

st.subheader("Daily PnL Time Series")
fig, ax = plt.subplots(figsize=(12,5))
sns.lineplot(data=df, x='date', y='daily_pnl', hue='classification', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Distribution of Key Metrics")
col1, col2 = st.columns(2)
with col1:
    fig2, ax2 = plt.subplots()
    df['daily_pnl'].hist(bins=50, ax=ax2)
    ax2.set_title('Daily PnL')
    st.pyplot(fig2)
with col2:
    fig3, ax3 = plt.subplots()
    df['win_rate'].hist(bins=30, ax=ax3)
    ax3.set_title('Win Rate')
    st.pyplot(fig3)