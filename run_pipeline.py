from src.data_loader import load_fear_greed_data, load_trader_data, merge_data, save_processed_data
from src.feature_engineering import compute_daily_metrics, add_trader_segments
from src.analysis import generate_all_comparisons, segment_sentiment_performance
from src.visualization import (plot_pnl_by_sentiment, plot_behavior_by_sentiment,
                               plot_segment_performance)
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def main():
    print("Step 1: Loading raw data...")
    fear = load_fear_greed_data()
    trader = load_trader_data()

    print("Step 2: Merging datasets...")
    merged = merge_data(trader, fear)
    save_processed_data(merged)

    print("Step 3: Computing daily metrics and segments...")
    daily = compute_daily_metrics(merged)
    daily = add_trader_segments(daily)
    daily.to_parquet(PROJECT_ROOT / "data" / "processed" / "daily_metrics.parquet", index=False)

    print("Step 4: Running statistical analysis...")
    comp_df = generate_all_comparisons(daily)
    comp_df.to_csv(PROJECT_ROOT / "reports" / "tables" / "sentiment_comparison.csv", index=False)

    seg_perf = segment_sentiment_performance(daily, 'leverage_segment', 'daily_pnl')
    seg_perf.to_csv(PROJECT_ROOT / "reports" / "tables" / "leverage_segment_performance.csv", index=False)

    behavior = daily.groupby('classification')[['avg_leverage', 'trade_count', 'long_short_ratio']].mean()
    behavior.to_csv(PROJECT_ROOT / "reports" / "tables" / "behavior_by_sentiment.csv")

    print("Step 5: Generating visualizations...")
    plot_pnl_by_sentiment(daily)
    plot_behavior_by_sentiment(daily, 'avg_leverage', 'Average Leverage by Sentiment', 'Leverage')
    plot_behavior_by_sentiment(daily, 'trade_count', 'Trade Frequency by Sentiment', 'Number of trades')
    plot_behavior_by_sentiment(daily, 'long_short_ratio', 'Long/Short Ratio by Sentiment', 'Long/Short Ratio')

    seg_perf_plot = daily.groupby(['classification', 'leverage_segment'])['daily_pnl'].mean().reset_index()
    plot_segment_performance(seg_perf_plot, 'leverage_segment')

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()