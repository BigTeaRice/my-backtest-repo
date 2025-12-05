"""
RSI超买超卖策略
RSI低于30买入，高于70卖出
"""

from backtesting import Strategy
import pandas as pd
import numpy as np
import talib


class RSIStrategy(Strategy):
    """RSI超买超卖策略"""
    
    # 策略参数
    rsi_period = 14
    oversold = 30
    overbought = 70
    rsi_exit = 50  # RSI回到50时平仓
    
    def init(self):
        """初始化指标"""
        # 计算RSI
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
    
    def next(self):
        """每日交易逻辑"""
        current_rsi = self.rsi[-1]
        
        if pd.isna(current_rsi):
            return
        
        has_position = self.position.size > 0
        
        # 超卖信号：RSI低于oversold，买入
        if not has_position and current_rsi < self.oversold:
            size = int(self.equity * 0.5 / self.data.Close[-1])
            if size > 0:
                self.buy(size=size)
        
        # 超买信号：RSI高于overbought，卖出
        elif has_position and current_rsi > self.overbought:
            self.position.close()
        
        # 或者RSI回到中间值时平仓
        elif has_position and abs(current_rsi - self.rsi_exit) < 5:
            if (current_rsi > self.oversold and self.rsi[-2] <= self.oversold):
                self.position.close()
