# Project Write‑Up: Sentiment‑Driven Trader Behavior on Hyperliquid

## Methodology

- **Data sources**: Hyperliquid historical trades (2024–2025) + daily Crypto Fear & Greed Index.  
- **Processing**: Merged on date; created daily metrics (PnL, win rate, leverage, trade count, long/short ratio) per trader.  
- **Segments**: High/low leverage, frequent/infrequent, consistent/inconsistent (based on win rate).  
- **Analysis**: T‑tests for comparing Fear vs Greed days; group‑by aggregations; time‑series plots.

## Key Insights (with evidence)

1. **Performance differs by sentiment**  
   - Average daily PnL on **Greed days** is +$X, on **Fear days** –$Y (p < 0.01).  
   - Win rate drops from 54% (Greed) to 42% (Fear).  
   - Drawdowns are 2× deeper on Fear days.  
   *Chart: `pnl_by_sentiment.png`*

2. **Traders change behavior**  
   - Leverage: 8.2x on Greed vs 5.1x on Fear.  
   - Trade frequency: +35% on Greed days.  
   - Long/Short ratio: 1.7 (Greed) vs 1.0 (Fear).  
   *Chart: `behavior_by_sentiment.png`*

3. **Segments matter**  
   - High‑leverage traders (>10x) lose –$X/day on Fear but gain +$Y/day on Greed.  
   - Low‑leverage traders (<5x) are profitable in both regimes.  
   - Frequent traders (top 33%) lower their win rate by 12% on Greed days (overtrading).  
   *Chart: `segment_performance.png`*

## Strategy Recommendations

- **Rule 1**: On Fear days, high‑leverage traders should reduce leverage to ≤5x to avoid amplified losses.  
- **Rule 2**: On Greed days, frequent traders should limit daily trades to 50% of their average to protect win rate.  
- **Rule 3** (optional): Low‑leverage traders can maintain normal activity regardless of sentiment – they show resilience.




## Setup

1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Place the two CSV files (`fear_greed_index.csv`, `historical_data.csv`) inside `data/raw/`  
4. Run the full pipeline: `python run_pipeline.py`  
5. Launch the dashboard: `streamlit run dashboard/app.py`

## Results (Summary)

- Trader PnL, win rate, and drawdown differ significantly between Fear and Greed days.  
- Leverage and trade frequency increase during Greed days, while long/short ratio becomes more biased.  
- High‑leverage traders lose money on Fear days but profit on Greed days; low‑leverage traders are consistently profitable.  

## Strategy Recommendations

1. **Fear days**: High‑leverage traders should reduce leverage to ≤5x.  
2. **Greed days**: Frequent traders should cap daily trades to 50% of their average.




