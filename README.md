# Golden-Cross-Strategy
The Golden Cross is a powerful bullish signal occurring when a short-term moving average (typically the 50-day SMA) crosses above a long-term moving average (the 200-day SMA). It signals a definitive shift in momentum, suggesting a long-term uptrend is forming. Traders often view this as a "buy" signal confirmed by high trading volume.

📈 Overview
This repository contains reference implementations for the Golden Cross, a classic technical analysis pattern used to identify major bullish trend reversals.
Strategy Logic
A Golden Cross is identified when:
1.	The Short-Term Moving Average (Fast) is currently above the Long-Term Moving Average (Slow).
2.	On the previous candle, the Fast MA was below the Slow MA.

Implementation
🐍 Python (Pandas)
Perfect for algorithmic backtesting.

## Assuming 'df' is a DataFrame with 'sma50' and 'sma200'
```python
df['signal'] = (df['sma50'] > df['sma200']) & (df['sma50'].shift(1) <= df['sma200'].shift(1))
golden_crosses = df[df['signal'] == True]
```

🟦 TypeScript
Useful for real-time alerts or frontend charting.

```typescript
const isGoldenCross = (fast: number[], slow: number[]): boolean => {
  const len = fast.length;
  return fast[len - 1] > slow[len - 1] && fast[len - 2] <= slow[len - 2];
};
```

Key Technical Specs
• Short-term MA: 50 periods
• Long-term MA: 200 periods
• Confirmation: Often paired with Relative Strength Index (RSI) or Volume analysis to filter out "false" crosses in choppy markets.
