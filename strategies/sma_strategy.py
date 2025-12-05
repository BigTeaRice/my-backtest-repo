"""
SMA双均线策略
短期均线上穿长期均线买入，下穿卖出
"""

from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np


class SMAStrategy(Strategy):
    """SMA双均线策略"""
    
    # 策略参数
    n1 = 20  # 短期均线周期
    n2 = 50  # 长期均线周期
    
    def init(self):
        """初始化指标"""
        # 计算SMA
        self.sma_short = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), 
                               self.data.Close)
        self.sma_long = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), 
                              self.data.Close)
        
        # 记录信号
        self.buy_signal = self.I(lambda: np.zeros(len(self.data)), name='buy_signal')
        self.sell_signal = self.I(lambda: np.zeros(len(self.data)), name='sell_signal')
    
    def next(self):
        """每日交易逻辑"""
        current_idx = len(self.data) - 1
        
        # 确保有足够的数据
        if current_idx < max(self.n1, self.n2):
            return
        
        # 检查是否持有仓位
        has_position = self.position.size > 0
        
        # 金叉信号：短期上穿长期，买入
        if (not has_position and 
            crossover(self.sma_short, self.sma_long)):
            
            # 计算头寸大小（使用50%资金）
            size = int(self.equity * 0.5 / self.data.Close[-1])
            if size > 0:
                self.buy(size=size)
                self.buy_signal[-1] = 1
        
        # 死叉信号：短期下穿长期，卖出
        elif (has_position and 
              crossover(self.sma_long, self.sma_short)):
            
            self.position.close()
            self.sell_signal[-1] = 1
    
    def get_parameters(self):
        """返回策略参数"""
        return {
            'n1': self.n1,
            'n2': self.n2,
            'name': 'SMA双均线策略',
            'description': '短期均线上穿长期均线买入，下穿卖出'
        }
