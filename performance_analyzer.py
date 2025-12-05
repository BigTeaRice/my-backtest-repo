"""
性能分析模块
计算回测性能指标
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """性能分析器"""
    
    def calculate_all_metrics(self, 
                             df: pd.DataFrame, 
                             bt_output: dict,
                             initial_capital: float = 100000) -> Dict:
        """
        计算所有性能指标
        
        Args:
            df: 价格数据
            bt_output: 回测输出
            initial_capital: 初始资金
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        # 基本指标
        metrics['initial_capital'] = initial_capital
        metrics['final_value'] = bt_output.get('Equity Final [$]', initial_capital)
        metrics['total_return'] = (metrics['final_value'] / initial_capital) - 1
        
        # 计算年化收益率
        metrics['annual_return'] = self._calculate_annual_return(
            df.index[0], df.index[-1], metrics['total_return']
        )
        
        # 夏普比率
        metrics['sharpe_ratio'] = bt_output.get('Sharpe Ratio', 0)
        
        # 最大回撤
        metrics['max_drawdown'] = abs(bt_output.get('Max. Drawdown [%]', 0)) / 100
        
        # 索提诺比率
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(
            df, bt_output
        )
        
        # 交易统计
        metrics['total_trades'] = bt_output.get('# Trades', 0)
        metrics['win_rate'] = bt_output.get('Win Rate [%]', 0) / 100
        metrics['avg_win'] = bt_output.get('Avg. Trade [%]', 0) / 100
        
        # 风险指标
        metrics['volatility'] = self._calculate_volatility(df)
        metrics['beta'] = self._calculate_beta(df)
        
        # 时间分析
        metrics['start_date'] = df.index[0]
        metrics['end_date'] = df.index[-1]
        metrics['period_days'] = (df.index[-1] - df.index[0]).days
        
        # 计算月度收益率
        monthly_returns = self._calculate_monthly_returns(df)
        metrics['best_month'] = monthly_returns.max()
        metrics['worst_month'] = monthly_returns.min()
        
        # 盈亏比
        metrics['profit_factor'] = bt_output.get('Profit Factor', 0)
        
        # 连续盈亏
        metrics['max_consecutive_wins'] = bt_output.get('Max. Consecutive Wins', 0)
        metrics['max_consecutive_losses'] = bt_output.get('Max. Consecutive Losses', 0)
        
        return metrics
    
    def _calculate_annual_return(self, 
                               start_date, 
                               end_date, 
                               total_return: float) -> float:
        """计算年化收益率"""
        days = (end_date - start_date).days
        if days <= 0:
            return 0
        
        years = days / 365.25
        if years <= 0:
            return 0
        
        annual_return = (1 + total_return) ** (1 / years) - 1
        return annual_return
    
    def _calculate_sortino_ratio(self, 
                               df: pd.DataFrame, 
                               bt_output: dict) -> float:
        """计算索提诺比率"""
        returns = df['returns'].dropna()
        if len(returns) == 0:
            return 0
        
        target_return = 0
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0
        
        avg_return = returns.mean() * 252  # 年化
        sortino_ratio = (avg_return - 0.02) / (downside_std * np.sqrt(252))
        return sortino_ratio
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """计算波动率"""
        returns = df['returns'].dropna()
        if len(returns) == 0:
            return 0
        return returns.std() * np.sqrt(252)
    
    def _calculate_beta(self, df: pd.DataFrame, benchmark='^HSI') -> float:
        """计算贝塔值"""
        # 这里需要基准数据，简化实现
        try:
            returns = df['returns'].dropna()
            if len(returns) < 30:
                return 1.0
            
            # 简化计算
            covariance = returns.rolling(30).cov(returns.shift(1)).mean()
            variance = returns.var()
            
            if variance == 0:
                return 1.0
            
            beta = covariance / variance
            return beta
        except:
            return 1.0
    
    def _calculate_monthly_returns(self, df: pd.DataFrame) -> pd.Series:
        """计算月度收益率"""
        monthly_prices = df['close'].resample('M').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        return monthly_returns
    
    def generate_summary_table(self, metrics: Dict) -> pd.DataFrame:
        """生成汇总表格"""
        summary_data = {
            '指标': [
                '初始资金', '最终净值', '总收益率', '年化收益率',
                '夏普比率', '最大回撤', '波动率', '总交易次数',
                '胜率', '平均盈亏比', '索提诺比率', '盈亏因子'
            ],
            '数值': [
                f"{metrics.get('initial_capital', 0):,.2f}",
                f"{metrics.get('final_value', 0):,.2f}",
                f"{metrics.get('total_return', 0):.2%}",
                f"{metrics.get('annual_return', 0):.2%}",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('max_drawdown', 0):.2%}",
                f"{metrics.get('volatility', 0):.2%}",
                f"{metrics.get('total_trades', 0):,d}",
                f"{metrics.get('win_rate', 0):.2%}",
                f"{metrics.get('avg_win', 0):.2f}",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                f"{metrics.get('profit_factor', 0):.2f}"
            ],
            '说明': [
                '回测开始时的资金',
                '回测结束时的总资产',
                '整个回测期间的总回报',
                '年化后的收益率',
                '每单位风险获得的超额回报',
                '最大的资金回撤幅度',
                '价格的年化波动率',
                '总交易次数',
                '盈利交易的比例',
                '平均盈利与平均亏损之比',
                '只考虑下行风险的调整后收益',
                '总盈利与总亏损之比'
            ]
        }
        
        return pd.DataFrame(summary_data)
