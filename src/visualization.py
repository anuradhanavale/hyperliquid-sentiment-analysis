import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def set_style():
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_pnl_by_sentiment(df: pd.DataFrame):
    set_style()
    plt.figure()
    sns.boxplot(data=df, x='classification', y='daily_pnl')
    plt.title("Daily PnL Distribution by Market Sentiment")
    plt.savefig(FIGURES_DIR / "pnl_by_sentiment.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_behavior_by_sentiment(df: pd.DataFrame, metric: str, title: str, ylabel: str):
    set_style()
    plt.figure()
    sns.barplot(data=df, x='classification', y=metric)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(FIGURES_DIR / f"{metric}_by_sentiment.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_segment_performance(df: pd.DataFrame, segment_col: str, metric: str = 'daily_pnl'):
    set_style()
    plt.figure()
    sns.barplot(data=df, x='classification', y=metric, hue=segment_col)
    plt.title(f"{metric} by Sentiment and {segment_col}")
    plt.legend(title=segment_col)
    plt.savefig(FIGURES_DIR / f"{segment_col}_{metric}_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    daily = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "daily_metrics.parquet")

    plot_pnl_by_sentiment(daily)
    plot_behavior_by_sentiment(daily, 'avg_leverage', 'Average Leverage by Sentiment', 'Leverage')
    plot_behavior_by_sentiment(daily, 'trade_count', 'Trade Frequency by Sentiment', 'Number of trades')
    plot_behavior_by_sentiment(daily, 'long_short_ratio', 'Long/Short Ratio by Sentiment', 'Long/Short Ratio')

    seg_perf = daily.groupby(['classification', 'leverage_segment'])['daily_pnl'].mean().reset_index()
    plot_segment_performance(seg_perf, 'leverage_segment')
    print("All figures saved.")