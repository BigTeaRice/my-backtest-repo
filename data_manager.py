"""
数据管理模块
负责数据获取、清洗、存储
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import pickle
from typing import Optional, Tuple

warnings.filterwarnings('ignore')


class DataManager:
    """数据管理器"""
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_data(self, 
                   symbol: str, 
                   start_date: str, 
                   end_date: str,
                   interval: str = '1d',
                   use_cache: bool = True) -> pd.DataFrame:
        """
        获取数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            use_cache: 是否使用缓存
            
        Returns:
            pandas DataFrame
        """
        # 检查缓存
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if not cached_data.empty:
                    print(f"使用缓存数据: {cache_key}")
                    return cached_data
        
        try:
            # 使用yfinance获取数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"警告: {symbol} 数据为空")
                return df
            
            # 重命名列
            df.columns = [col.lower() for col in df.columns]
            
            # 确保有必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'close' and 'adj close' in df.columns:
                        df['close'] = df['adj close']
                    else:
                        df[col] = 0
            
            # 计算回报率
            df['returns'] = df['close'].pct_change()
            
            # 计算技术指标
            df = self._add_technical_indicators(df)
            
            # 清理数据
            df = df.dropna()
            
            # 保存到缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            print(f"获取数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 简单移动平均
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # 指数移动平均
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 随机指标KDJ
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        df['k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['d'] = df['k'].rolling(window=3).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # ATR (平均真实范围)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        return df
    
    def get_market_data(self, 
                       symbols: list, 
                       start_date: str, 
                       end_date: str) -> dict:
        """获取多个标的数据"""
        data_dict = {}
        for symbol in symbols:
            df = self.fetch_data(symbol, start_date, end_date)
            if not df.empty:
                data_dict[symbol] = df
        return data_dict
    
    def update_all_data(self):
        """更新所有数据"""
        # 这里可以添加定期数据更新逻辑
        pass
