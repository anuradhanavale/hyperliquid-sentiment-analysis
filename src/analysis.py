import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_TABLES = PROJECT_ROOT / "reports" / "tables"

def load_daily_metrics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "daily_metrics.parquet")

def sentiment_comparison(df: pd.DataFrame, metric: str) -> dict:
    fear = df[df['classification'] == 'Fear'][metric].dropna()
    greed = df[df['classification'] == 'Greed'][metric].dropna()
    t_stat, p_val = stats.ttest_ind(fear, greed, equal_var=False)
    return {
        'metric': metric,
        'fear_mean': fear.mean(),
        'greed_mean': greed.mean(),
        'fear_median': fear.median(),
        'greed_median': greed.median(),
        'p_value': p_val,
        'significant': p_val < 0.05
    }

def segment_sentiment_performance(df: pd.DataFrame, segment_col: str, metric: str = 'daily_pnl') -> pd.DataFrame:
    return df.groupby(['classification', segment_col])[metric].mean().reset_index()

def generate_all_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ['daily_pnl', 'win_rate', 'avg_leverage', 'trade_count', 'long_short_ratio', 'max_daily_drawdown']
    results = []
    for metric in metrics:
        if metric in df.columns:
            results.append(sentiment_comparison(df, metric))
    return pd.DataFrame(results)

if __name__ == "__main__":
    REPORTS_TABLES.mkdir(exist_ok=True, parents=True)
    daily = load_daily_metrics()

    comp_df = generate_all_comparisons(daily)
    comp_df.to_csv(REPORTS_TABLES / "sentiment_comparison.csv", index=False)

    # Leverage segment performance
    seg_perf = segment_sentiment_performance(daily, 'leverage_segment', 'daily_pnl')
    seg_perf.to_csv(REPORTS_TABLES / "leverage_segment_performance.csv", index=False)

    # Behavior summary
    behavior = daily.groupby('classification')[['avg_leverage', 'trade_count', 'long_short_ratio']].mean()
    behavior.to_csv(REPORTS_TABLES / "behavior_by_sentiment.csv")

    print("Analysis tables saved.")