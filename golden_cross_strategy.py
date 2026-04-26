"""
Golden Cross SMA Strategy
A trend-following strategy using 50-day and 200-day Simple Moving Averages.

Entry: Buy when 50-day SMA crosses above 200-day SMA (Golden Cross)
Exit: Sell when 50-day SMA crosses below 200-day SMA (Death Cross)

Author: Agentic Trading ML
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    """Trading signal data class"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Trade:
    """Trade record data class"""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_days: int = 0


class GoldenCrossStrategy:
    """
    Golden Cross SMA Strategy
    
    This strategy generates buy signals when the 50-day Simple Moving Average 
    crosses above the 200-day SMA (Golden Cross), and sell signals when the 
    50-day SMA crosses below the 200-day SMA (Death Cross).
    
    Parameters:
    -----------
    fast_period : int
        Period for fast moving average (default: 50)
    slow_period : int
        Period for slow moving average (default: 200)
    stop_loss_pct : float
        Stop loss percentage (default: 0.10)
    position_size_pct : float
        Position size as percentage of equity (default: 0.20)
    max_positions : int
        Maximum number of concurrent positions (default: 5)
    
    Example:
    --------
    >>> import pandas as pd
    >>> from golden_cross_strategy import GoldenCrossStrategy
    >>> 
    >>> # Load historical data
    >>> df = pd.read_csv('spy_data.csv', parse_dates=['date'])
    >>> df.set_index('date', inplace=True)
    >>> 
    >>> # Initialize strategy
    >>> strategy = GoldenCrossStrategy(
    ...     fast_period=50,
    ...     slow_period=200,
    ...     stop_loss_pct=0.10,
    ...     position_size_pct=0.20
    ... )
    >>> 
    >>> # Generate signals
    >>> signals = strategy.generate_signals(df)
    >>> 
    >>> # Run backtest
    >>> results = strategy.backtest(df, initial_capital=100000)
    >>> print(f"Total Return: {results['total_return']:.2%}")
    >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    """
    
    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        stop_loss_pct: float = 0.10,
        take_profit_pct: Optional[float] = None,
        position_size_pct: float = 0.20,
        max_positions: int = 5,
        min_volume: int = 1_000_000,
        min_price: float = 10.0
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.min_volume = min_volume
        self.min_price = min_price
        
        self.positions: Dict[str, Dict] = {}
        self.signals: List[Signal] = []
        self.trades: List[Trade] = []
        
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period, min_periods=period).mean()
    
    def detect_crossover(
        self, 
        fast_ma: pd.Series, 
        slow_ma: pd.Series
    ) -> pd.Series:
        """
        Detect when fast MA crosses above slow MA (Golden Cross)
        
        Returns:
        --------
        pd.Series : Boolean series where True indicates a golden cross
        """
        # Previous day values
        fast_prev = fast_ma.shift(1)
        slow_prev = slow_ma.shift(1)
        
        # Golden cross: fast was below slow yesterday, above today
        golden_cross = (fast_prev < slow_prev) & (fast_ma > slow_ma)
        
        return golden_cross
    
    def detect_crossunder(
        self, 
        fast_ma: pd.Series, 
        slow_ma: pd.Series
    ) -> pd.Series:
        """
        Detect when fast MA crosses below slow MA (Death Cross)
        
        Returns:
        --------
        pd.Series : Boolean series where True indicates a death cross
        """
        # Previous day values
        fast_prev = fast_ma.shift(1)
        slow_prev = slow_ma.shift(1)
        
        # Death cross: fast was above slow yesterday, below today
        death_cross = (fast_prev > slow_prev) & (fast_ma < slow_ma)
        
        return death_cross
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the given price data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'
            Index should be datetime
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - 'sma_fast': Fast moving average
            - 'sma_slow': Slow moving average
            - 'golden_cross': Golden cross signals (1 = buy)
            - 'death_cross': Death cross signals (-1 = sell)
            - 'signal': Combined signal (1 = buy, -1 = sell, 0 = hold)
        """
        df = data.copy()
        
        # Calculate moving averages
        df['sma_fast'] = self.calculate_sma(df['close'], self.fast_period)
        df['sma_slow'] = self.calculate_sma(df['close'], self.slow_period)
        
        # Detect crossovers
        df['golden_cross'] = self.detect_crossover(df['sma_fast'], df['sma_slow'])
        df['death_cross'] = self.detect_crossunder(df['sma_fast'], df['sma_slow'])
        
        # Combined signal
        df['signal'] = 0
        df.loc[df['golden_cross'], 'signal'] = 1  # Buy signal
        df.loc[df['death_cross'], 'signal'] = -1  # Sell signal
        
        # Apply filters
        if 'volume' in df.columns:
            # Only trade if volume > minimum
            low_volume = df['volume'] < self.min_volume
            df.loc[low_volume, 'signal'] = 0
        
        # Only trade if price > minimum
        low_price = df['close'] < self.min_price
        df.loc[low_price, 'signal'] = 0
        
        return df
    
    def backtest(
        self, 
        data: pd.DataFrame, 
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.001
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with OHLCV columns
        initial_capital : float
            Starting capital (default: 100000)
        commission : float
            Commission per trade as decimal (default: 0.001 = 0.1%)
        slippage : float
            Slippage per trade as decimal (default: 0.001 = 0.1%)
            
        Returns:
        --------
        Dict : Backtest results including:
            - total_return: Total return percentage
            - annualized_return: Annualized return
            - sharpe_ratio: Sharpe ratio
            - max_drawdown: Maximum drawdown
            - win_rate: Win rate percentage
            - total_trades: Number of trades
            - equity_curve: Daily portfolio values
            - trades: List of all trades
        """
        # Generate signals
        df = self.generate_signals(data)
        
        # Initialize tracking variables
        capital = initial_capital
        equity_curve = []
        trades = []
        position = None
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if pd.isna(row['sma_fast']) or pd.isna(row['sma_slow']):
                # Skip until we have enough data for MAs
                equity_curve.append({
                    'date': timestamp,
                    'equity': capital,
                    'drawdown': 0
                })
                continue
            
            # Check for entry signal
            if row['signal'] == 1 and position is None:
                # Calculate position size
                position_value = capital * self.position_size_pct
                shares = position_value / row['close']
                
                # Apply slippage and commission
                entry_price = row['close'] * (1 + slippage)
                commission_cost = position_value * commission
                
                position = {
                    'entry_date': timestamp,
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': entry_price * (1 - self.stop_loss_pct),
                    'take_profit': entry_price * (1 + self.take_profit_pct) if self.take_profit_pct else None
                }
                
                capital -= commission_cost
            
            # Check for exit conditions
            elif position is not None:
                current_price = row['close']
                
                # Check stop loss
                exit_reason = None
                exit_price = current_price
                
                if current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                # Check take profit
                elif position['take_profit'] and current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = position['take_profit']
                # Check death cross signal
                elif row['signal'] == -1:
                    exit_reason = 'death_cross'
                    exit_price = current_price * (1 - slippage)
                
                if exit_reason:
                    # Close position
                    position_value = position['shares'] * exit_price
                    commission_cost = position_value * commission
                    
                    # Calculate P&L
                    gross_pnl = position['shares'] * (exit_price - position['entry_price'])
                    net_pnl = gross_pnl - commission_cost * 2  # Entry + exit commission
                    
                    trade = Trade(
                        entry_date=position['entry_date'],
                        exit_date=timestamp,
                        symbol='UNKNOWN',  # Should be passed as parameter
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['shares'],
                        pnl=net_pnl,
                        pnl_pct=(exit_price - position['entry_price']) / position['entry_price'],
                        duration_days=(timestamp - position['entry_date']).days
                    )
                    trades.append(trade)
                    
                    capital += net_pnl
                    position = None
            
            # Calculate current equity
            if position:
                position_value = position['shares'] * row['close']
                current_equity = capital + position_value
            else:
                current_equity = capital
            
            # Calculate drawdown
            peak_equity = max([e['equity'] for e in equity_curve]) if equity_curve else current_equity
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            
            equity_curve.append({
                'date': timestamp,
                'equity': current_equity,
                'drawdown': drawdown
            })
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        total_return = (equity_df['equity'].iloc[-1] - initial_capital) / initial_capital
        
        # Calculate daily returns for Sharpe
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        avg_return = equity_df['daily_return'].mean()
        std_return = equity_df['daily_return'].std()
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        max_drawdown = equity_df['drawdown'].max()
        
        # Calculate win rate
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return,  # Simplified - should adjust for time period
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades
        }


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    print("Golden Cross SMA Strategy")
    print("=" * 50)
    print()
    print("This strategy generates signals based on moving average crossovers.")
    print()
    print("Usage:")
    print("  from golden_cross_strategy import GoldenCrossStrategy")
    print("  ")
    print("  strategy = GoldenCrossStrategy(")
    print("      fast_period=50,")
    print("      slow_period=200,")
    print("      stop_loss_pct=0.10")
    print("  )")
    print("  ")
    print("  results = strategy.backtest(data, initial_capital=100000)")
    print()
    print("For detailed documentation, see README.md")
