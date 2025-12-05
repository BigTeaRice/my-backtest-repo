#!/usr/bin/env python3
"""
多策略回测系统主程序
支持：SMA, RSI, MACD, 布林带, KDI 五种技术指标策略
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 策略模块
from strategies.sma_strategy import SMAStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_strategy import BollingerStrategy
from strategies.kdi_strategy import KDIStrategy

# 回测引擎
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 数据管理
from data_manager import DataManager
from performance_analyzer import PerformanceAnalyzer
from report_generator import ReportGenerator


class MultiStrategyBacktestSystem:
    """多策略回测系统主类"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analyzer = PerformanceAnalyzer()
        self.report_generator = ReportGenerator()
        self.strategies = {
            'SMA': SMAStrategy,
            'RSI': RSIStrategy,
            'MACD': MACDStrategy,
            'Bollinger': BollingerStrategy,
            'KDI': KDIStrategy
        }
        
        # 支持的标的
        self.supported_symbols = {
            'HSI': {'name': '恒生指数', 'yfinance': '^HSI'},
            'SPY': {'name': '标普500', 'yfinance': 'SPY'},
            'BTC-USD': {'name': '比特币', 'yfinance': 'BTC-USD'},
            'AAPL': {'name': '苹果公司', 'yfinance': 'AAPL'},
            '000001.SS': {'name': '上证指数', 'yfinance': '000001.SS'},
        }
    
    def run_backtest(self, 
                    strategy_name: str,
                    symbol: str,
                    start_date: str = '2020-01-01',
                    end_date: str = None,
                    initial_capital: float = 100000,
                    commission: float = 0.001,
                    **strategy_params) -> Dict:
        """
        运行单策略回测
        
        Args:
            strategy_name: 策略名称
            symbol: 交易标的
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            commission: 手续费率
            **strategy_params: 策略参数
            
        Returns:
            回测结果字典
        """
        # 设置结束日期为昨天
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"开始回测: {strategy_name} - {symbol}")
        print(f"时间范围: {start_date} 至 {end_date}")
        print(f"初始资金: {initial_capital}")
        
        # 1. 获取数据
        try:
            if symbol in self.supported_symbols:
                yf_symbol = self.supported_symbols[symbol]['yfinance']
                df = self.data_manager.fetch_data(yf_symbol, start_date, end_date)
            else:
                df = self.data_manager.fetch_data(symbol, start_date, end_date)
        except Exception as e:
            print(f"数据获取失败: {e}")
            return None
        
        if df.empty:
            print("获取的数据为空")
            return None
            
        print(f"获取数据 {len(df)} 条")
        
        # 2. 运行回测
        try:
            # 选择策略类
            strategy_class = self.strategies.get(strategy_name)
            if not strategy_class:
                raise ValueError(f"不支持策略: {strategy_name}")
            
            # 创建回测实例
            bt = Backtest(
                df,
                strategy_class,
                cash=initial_capital,
                commission=commission,
                exclusive_orders=True
            )
            
            # 运行回测
            output = bt.run(**strategy_params)
            
            # 3. 分析性能
            performance_stats = self.analyzer.calculate_all_metrics(
                df, 
                output, 
                initial_capital
            )
            
            # 4. 生成报告
            report_data = self.report_generator.generate_report(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                performance_stats=performance_stats,
                trades=output['_trades'] if '_trades' in output else None,
                equity_curve=bt._equity_curve
            )
            
            # 5. 保存结果
            self._save_results(report_data, strategy_name, symbol)
            
            return report_data
            
        except Exception as e:
            print(f"回测运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_comparison(self,
                      symbols: List[str],
                      start_date: str = '2020-01-01',
                      end_date: str = None) -> pd.DataFrame:
        """
        运行多策略对比回测
        
        Args:
            symbols: 标的列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            策略对比DataFrame
        """
        comparison_results = []
        
        for symbol in symbols:
            for strategy_name in self.strategies.keys():
                try:
                    result = self.run_backtest(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=100000
                    )
                    
                    if result and 'performance_stats' in result:
                        stats = result['performance_stats']
                        comparison_results.append({
                            'strategy': strategy_name,
                            'symbol': symbol,
                            'total_return': stats.get('total_return', 0),
                            'annual_return': stats.get('annual_return', 0),
                            'sharpe_ratio': stats.get('sharpe_ratio', 0),
                            'max_drawdown': stats.get('max_drawdown', 0),
                            'win_rate': stats.get('win_rate', 0),
                            'total_trades': stats.get('total_trades', 0)
                        })
                        
                except Exception as e:
                    print(f"回测 {strategy_name} - {symbol} 失败: {e}")
        
        # 创建对比DataFrame
        df_comparison = pd.DataFrame(comparison_results)
        
        # 保存到CSV
        csv_path = 'public/strategy_comparison.csv'
        df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"策略对比结果已保存到: {csv_path}")
        
        return df_comparison
    
    def _save_results(self, report_data: Dict, strategy_name: str, symbol: str):
        """保存回测结果"""
        # 创建报告目录
        os.makedirs('public/reports', exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_{symbol}_{timestamp}"
        
        # 保存JSON报告
        json_path = f"public/reports/{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 保存HTML报告
        html_path = f"public/reports/{filename}.html"
        self.report_generator.save_html_report(report_data, html_path)
        
        print(f"回测报告已保存:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
    
    def run_from_config(self, config_file: str = 'config.json'):
        """从配置文件运行回测"""
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在")
            return
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 运行配置的回测
        for backtest_config in config.get('backtests', []):
            self.run_backtest(**backtest_config)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多策略回测系统')
    parser.add_argument('--strategy', type=str, default='SMA', 
                       choices=['SMA', 'RSI', 'MACD', 'Bollinger', 'KDI'],
                       help='策略名称')
    parser.add_argument('--symbol', type=str, default='HSI', 
                       help='交易标的')
    parser.add_argument('--start', type=str, default='2020-01-01',
                       help='开始日期')
    parser.add_argument('--end', type=str, default=None,
                       help='结束日期')
    parser.add_argument('--capital', type=float, default=100000,
                       help='初始资金')
    parser.add_argument('--compare', action='store_true',
                       help='运行策略对比')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建回测系统实例
    system = MultiStrategyBacktestSystem()
    
    if args.config:
        # 从配置文件运行
        system.run_from_config(args.config)
    elif args.compare:
        # 运行策略对比
        symbols = ['HSI', 'SPY', 'BTC-USD']
        system.run_comparison(symbols, args.start, args.end)
    else:
        # 运行单策略回测
        result = system.run_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital
        )
        
        if result:
            print("\n" + "="*50)
            print("回测完成!")
            print("="*50)
            stats = result['performance_stats']
            print(f"总收益率: {stats.get('total_return', 0):.2%}")
            print(f"年化收益率: {stats.get('annual_return', 0):.2%}")
            print(f"夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
            print(f"最大回撤: {stats.get('max_drawdown', 0):.2%}")
            print(f"胜率: {stats.get('win_rate', 0):.2%}")


if __name__ == "__main__":
    main()
