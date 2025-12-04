#!/usr/bin/env python3
# main.py - 多策略回测系统（SMA, RSI, MACD, 布林带, KDJ）

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. 配置参数
# ------------------------------------------------------------------
CONFIG = {
    # 股票标的（简化为常用标的）
    "STOCKS": {
        "^HSI": "恒生指数",
        "0700.HK": "腾讯控股", 
        "9988.HK": "阿里巴巴",
        "AAPL": "苹果",
        "MSFT": "微软",
        "GOOGL": "谷歌",
        "TSLA": "特斯拉",
        "NVDA": "英伟达",
        "SPY": "标普500 ETF",
        "QQQ": "纳指100 ETF"
    },
    
    # 回测参数
    "BACKTEST": {
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "initial_cash": 100000,
        "commission": 0.002,
    },
    
    # 策略参数
    "STRATEGY_PARAMS": {
        "SMA": {"fast": 10, "slow": 30},
        "RSI": {"period": 14, "oversold": 30, "overbought": 70},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BB": {"period": 20, "std_dev": 2},
        "KDJ": {"period": 9, "k_period": 3, "d_period": 3}
    }
}

# ------------------------------------------------------------------
# 2. 技术指标计算函数
# ------------------------------------------------------------------
def calculate_sma(series, period):
    """计算简单移动平均线"""
    return series.rolling(window=period).mean()

def calculate_ema(series, period):
    """计算指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    """计算RSI指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(series, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """计算布林带"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """计算KDJ指标（随机指标）"""
    # 计算%K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # 避免除零错误
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, 1)  # 将0替换为1
    
    k_value = 100 * ((close - lowest_low) / denominator)
    
    # 计算%D（K值的移动平均）
    d_value = k_value.rolling(window=d_period).mean()
    
    # 计算%J
    j_value = 3 * k_value - 2 * d_value
    
    return k_value.fillna(50), d_value.fillna(50), j_value.fillna(50)

# ------------------------------------------------------------------
# 3. 策略定义
# ------------------------------------------------------------------
class SmaStrategy(Strategy):
    """SMA双均线策略"""
    Name = "SMA策略"
    
    def init(self):
        params = CONFIG["STRATEGY_PARAMS"]["SMA"]
        self.fast_period = params["fast"]
        self.slow_period = params["slow"]
        
        # 计算均线
        self.sma_fast = self.I(calculate_sma, self.data.Close, self.fast_period)
        self.sma_slow = self.I(calculate_sma, self.data.Close, self.slow_period)
    
    def next(self):
        # 快速均线上穿慢速均线 - 买入
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy()
        
        # 快速均线下穿慢速均线 - 卖出
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()

class RsiStrategy(Strategy):
    """RSI超买超卖策略"""
    Name = "RSI策略"
    
    def init(self):
        params = CONFIG["STRATEGY_PARAMS"]["RSI"]
        self.period = params["period"]
        self.oversold = params["oversold"]
        self.overbought = params["overbought"]
        
        # 计算RSI
        self.rsi = self.I(calculate_rsi, self.data.Close, self.period)
    
    def next(self):
        current_rsi = self.rsi[-1]
        
        # RSI低于超卖线 - 买入
        if current_rsi < self.oversold and not self.position:
            self.buy()
        
        # RSI高于超买线 - 卖出
        elif current_rsi > self.overbought and self.position:
            self.position.close()

class MacdStrategy(Strategy):
    """MACD交叉策略"""
    Name = "MACD策略"
    
    def init(self):
        params = CONFIG["STRATEGY_PARAMS"]["MACD"]
        self.fast = params["fast"]
        self.slow = params["slow"]
        self.signal = params["signal"]
        
        # 计算MACD
        self.macd, self.signal_line, self.histogram = self.I(
            calculate_macd, pd.Series(self.data.Close), 
            self.fast, self.slow, self.signal
        )
    
    def next(self):
        # MACD线上穿信号线 - 买入
        if crossover(self.macd, self.signal_line):
            if not self.position:
                self.buy()
        
        # MACD线下穿信号线 - 卖出
        elif crossover(self.signal_line, self.macd):
            if self.position:
                self.position.close()

class BollingerBandsStrategy(Strategy):
    """布林带策略"""
    Name = "布林带策略"
    
    def init(self):
        params = CONFIG["STRATEGY_PARAMS"]["BB"]
        self.period = params["period"]
        self.std_dev = params["std_dev"]
        
        # 计算布林带
        self.upper, self.middle, self.lower = self.I(
            calculate_bollinger_bands, pd.Series(self.data.Close),
            self.period, self.std_dev
        )
    
    def next(self):
        current_price = self.data.Close[-1]
        
        # 价格跌破下轨 - 买入
        if current_price < self.lower[-1] and not self.position:
            self.buy()
        
        # 价格突破上轨 - 卖出
        elif current_price > self.upper[-1] and self.position:
            self.position.close()

class KdjStrategy(Strategy):
    """KDJ策略"""
    Name = "KDJ策略"
    
    def init(self):
        params = CONFIG["STRATEGY_PARAMS"]["KDJ"]
        self.period = params["period"]
        self.k_period = params["k_period"]
        self.d_period = params["d_period"]
        
        # 计算KDJ
        self.k, self.d, self.j = self.I(
            calculate_stochastic, 
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Close),
            self.period, self.d_period
        )
    
    def next(self):
        # K线上穿D线且在超卖区 - 买入
        if (crossover(self.k, self.d) and 
            self.k[-1] < 20 and not self.position):
            self.buy()
        
        # K线下穿D线且在超买区 - 卖出
        elif (crossover(self.d, self.k) and 
              self.k[-1] > 80 and self.position):
            self.position.close()

# ------------------------------------------------------------------
# 4. 数据获取函数
# ------------------------------------------------------------------
def download_stock_data(ticker, start_date, end_date):
    """下载股票数据"""
    try:
        print(f"📥 下载 {ticker}...", end="")
        
        # 使用yfinance下载数据
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        if df.empty:
            print(f" ❌ 无数据")
            return None
        
        # 清理列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 重命名列
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})
        
        # 确保有必要的列
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print(f" ❌ 缺少必要列")
            return None
        
        # 清理数据
        df = df.dropna()
        
        if len(df) < 30:
            print(f" ❌ 数据不足")
            return None
        
        print(f" ✅ {len(df)}条")
        return df
        
    except Exception as e:
        print(f" ❌ 错误: {str(e)[:50]}")
        return None

# ------------------------------------------------------------------
# 5. 主回测函数
# ------------------------------------------------------------------
def run_strategy_backtest(strategy_class, ticker, stock_name, config):
    """运行单个策略回测"""
    try:
        # 下载数据
        df = download_stock_data(
            ticker,
            config["BACKTEST"]["start_date"],
            config["BACKTEST"]["end_date"]
        )
        
        if df is None:
            return None
        
        # 创建回测实例
        bt = Backtest(
            df,
            strategy_class,
            cash=config["BACKTEST"]["initial_cash"],
            commission=config["BACKTEST"]["commission"]
        )
        
        # 运行回测
        stats = bt.run()
        
        # 生成报告文件名
        safe_ticker = ticker.replace("^", "").replace(".", "_").replace("-", "_")
        filename = f"{strategy_class.Name}_{safe_ticker}.html"
        
        # 生成图表（简化版，避免bokeh问题）
        try:
            bt.plot(
                filename=f"public/reports/{filename}",
                open_browser=False,
                plot_volume=False,
                plot_drawdown=True
            )
        except Exception as plot_error:
            print(f"   ⚠️  图表生成失败: {plot_error}")
            # 继续处理
        
        # 准备统计数据
        stats_dict = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool)) and not key.startswith('_'):
                stats_dict[key] = value
        
        # 添加额外信息
        stats_dict.update({
            "标的名称": stock_name,
            "标的代码": ticker,
            "策略名称": strategy_class.Name,
            "数据起点": str(df.index[0].date()),
            "数据终点": str(df.index[-1].date()),
            "数据条数": len(df),
            "初始资金": config["BACKTEST"]["initial_cash"],
            "手续费率": config["BACKTEST"]["commission"],
        })
        
        return {
            "file": f"reports/{filename}",
            "stats": stats_dict
        }
        
    except Exception as e:
        print(f"   ❌ 回测失败: {str(e)[:50]}")
        return None

# ------------------------------------------------------------------
# 6. 主程序
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("📊 多策略回测系统")
    print("=" * 60)
    print("策略列表: SMA, RSI, MACD, 布林带, KDJ")
    print(f"标的数量: {len(CONFIG['STOCKS'])}")
    print(f"回测期间: {CONFIG['BACKTEST']['start_date']} 到 {CONFIG['BACKTEST']['end_date']}")
    print()
    
    # 创建输出目录
    os.makedirs("public/reports", exist_ok=True)
    
    # 策略列表
    strategies = [
        SmaStrategy,
        RsiStrategy,
        MacdStrategy,
        BollingerBandsStrategy,
        KdjStrategy
    ]
    
    # 存储结果
    results = {}
    all_reports = []
    
    total_combinations = len(strategies) * len(CONFIG["STOCKS"])
    completed = 0
    
    for strategy_class in strategies:
        strategy_name = strategy_class.Name
        results[strategy_name] = {}
        
        print(f"\n📈 运行策略: {strategy_name}")
        print("-" * 40)
        
        for ticker, stock_name in CONFIG["STOCKS"].items():
            print(f"  {stock_name} ({ticker})...", end="")
            
            # 运行回测
            result = run_strategy_backtest(strategy_class, ticker, stock_name, CONFIG)
            
            if result:
                results[strategy_name][ticker] = result
                
                # 添加到报告列表
                all_reports.append({
                    "策略": strategy_name,
                    "标的代码": ticker,
                    "标的名称": stock_name,
                    "年化收益%": result["stats"].get("Return (Ann.) [%]", 0),
                    "夏普比率": result["stats"].get("Sharpe Ratio", 0),
                    "最大回撤%": result["stats"].get("Max. Drawdown [%]", 0),
                    "总收益率%": result["stats"].get("Return [%]", 0),
                    "胜率%": result["stats"].get("Win Rate [%]", 0),
                    "交易次数": result["stats"].get("# Trades", 0),
                    "盈利因子": result["stats"].get("Profit Factor", 0),
                    "报告文件": result["file"],
                })
                
                completed += 1
                trades = result["stats"].get("# Trades", 0)
                returns = result["stats"].get("Return [%]", 0)
                print(f" ✅ {trades}笔交易, {returns:.1f}%")
            else:
                print(f" ❌ 失败")
    
    print(f"\n🎉 回测完成: {completed}/{total_combinations} 个组合")
    
    # 生成CSV报告
    if all_reports:
        df_reports = pd.DataFrame(all_reports)
        df_reports = df_reports.sort_values("夏普比率", ascending=False)
        df_reports.to_csv("public/strategy_comparison.csv", index=False, encoding='utf-8-sig')
        print(f"📊 CSV报告已生成: public/strategy_comparison.csv")
    
    # 生成HTML主页面
    generate_html(results, "public")
    
    print("\n" + "=" * 60)
    print("✅ 所有任务完成!")
    print("📁 输出目录: public/")
    print("🌐 请打开 public/index.html 查看结果")
    print("=" * 60)
    
    return True

def generate_html(results, output_dir):
    """生成HTML主页面"""
    
    # 构建策略选项
    strategy_options = ""
    for strategy_name in results.keys():
        strategy_options += f'<option value="{strategy_name}">{strategy_name}</option>\n'
    
    # 构建股票选项
    stock_options = ""
    for ticker, name in CONFIG["STOCKS"].items():
        stock_options += f'<option value="{ticker}">{name} ({ticker})</option>\n'
    
    # 最佳策略推荐
    recommendations = ""
    try:
        df = pd.read_csv("public/strategy_comparison.csv")
        if not df.empty:
            best = df.iloc[0]
            recommendations = f"""
            <div class="recommendations">
                <h3>🏆 最佳策略推荐</h3>
                <div class="rec-card">
                    <h4>{best['策略']} + {best['标的名称']}</h4>
                    <p>夏普比率: <strong>{best['夏普比率']:.2f}</strong></p>
                    <p>年化收益: <strong>{best['年化收益%']:.1f}%</strong></p>
                    <p>最大回撤: <strong>{best['最大回撤%']:.1f}%</strong></p>
                </div>
            </div>
            """
    except:
        pass
    
    # 转换为JSON
    results_json = json.dumps(results, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多策略回测系统</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
            margin-bottom: 15px;
        }}
        .strategy-tags {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }}
        .strategy-tag {{
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        .controls {{
            padding: 25px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
        }}
        .control-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .control-group {{
            flex: 1;
            min-width: 300px;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }}
        select {{
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            background: white;
        }}
        select:focus {{
            border-color: #26d0ce;
            outline: none;
        }}
        .btn-group {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }}
        .btn {{
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            color: white;
        }}
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(26, 41, 128, 0.3);
        }}
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        .btn-secondary:hover {{
            background: #5a6268;
        }}
        .content {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            padding: 25px;
            min-height: 700px;
        }}
        .chart-container {{
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            background: white;
        }}
        .chart-frame {{
            width: 100%;
            height: 700px;
            border: none;
        }}
        .stats-sidebar {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            max-height: 700px;
        }}
        .stats-title {{
            color: #1a2980;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #26d0ce;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
        }}
        .stats-table th, .stats-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .stats-table th {{
            background: #e9ecef;
            font-weight: 600;
        }}
        .stats-table tr:hover {{
            background: #f1f3f5;
        }}
        {recommendations}
        .footer {{
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
            background: #f8f9fa;
        }}
        @media (max-width: 1024px) {{
            .content {{ grid-template-columns: 1fr; }}
            .chart-frame {{ height: 500px; }}
            .stats-sidebar {{ max-height: 500px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 多策略回测系统</h1>
            <p>支持SMA, RSI, MACD, 布林带, KDJ 五种技术指标策略</p>
            <div class="strategy-tags">
                <span class="strategy-tag">SMA双均线</span>
                <span class="strategy-tag">RSI超买超卖</span>
                <span class="strategy-tag">MACD交叉</span>
                <span class="strategy-tag">布林带</span>
                <span class="strategy-tag">KDJ随机指标</span>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-row">
                <div class="control-group">
                    <label>选择策略:</label>
                    <select id="strategy-select">
                        {strategy_options}
                    </select>
                </div>
                <div class="control-group">
                    <label>选择标的:</label>
                    <select id="stock-select">
                        {stock_options}
                    </select>
                </div>
            </div>
            <div class="btn-group">
                <button class="btn btn-primary" onclick="loadReport()">
                    <span>📈</span> 加载回测报告
                </button>
                <button class="btn btn-secondary" onclick="downloadCSV()">
                    <span>📥</span> 下载完整报告
                </button>
            </div>
        </div>
        
        <div class="content">
            <div class="chart-container">
                <iframe id="chart-frame" class="chart-frame" 
                        title="回测图表"
                        src="about:blank">
                </iframe>
            </div>
            
            <div class="stats-sidebar">
                <h2 class="stats-title">📊 性能指标</h2>
                <table class="stats-table" id="stats-table">
                    <tbody>
                        <tr><td>标的名称</td><td id="stat-name">--</td></tr>
                        <tr><td>数据期间</td><td id="stat-period">--</td></tr>
                        <tr><td>数据条数</td><td id="stat-count">--</td></tr>
                        <tr><td>初始资金</td><td id="stat-initial">--</td></tr>
                        <tr><td>最终净值</td><td id="stat-final">--</td></tr>
                        <tr><td>总收益率</td><td id="stat-return">--</td></tr>
                        <tr><td>年化收益率</td><td id="stat-annual">--</td></tr>
                        <tr><td>最大回撤</td><td id="stat-drawdown">--</td></tr>
                        <tr><td>夏普比率</td><td id="stat-sharpe">--</td></tr>
                        <tr><td>交易次数</td><td id="stat-trades">--</td></tr>
                        <tr><td>胜率</td><td id="stat-winrate">--</td></tr>
                        <tr><td>盈利因子</td><td id="stat-profit">--</td></tr>
                    </tbody>
                </table>
                {recommendations if recommendations else ''}
            </div>
        </div>
        
        <div class="footer">
            <p>数据来源: Yahoo Finance | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>风险提示: 回测结果基于历史数据，不代表未来表现</p>
        </div>
    </div>
    
    <script>
        // 回测结果数据
        const RESULTS = {results_json};
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {{
            if (Object.keys(RESULTS).length > 0) {{
                const firstStrategy = Object.keys(RESULTS)[0];
                const firstTicker = Object.keys(RESULTS[firstStrategy])[0];
                
                if (firstStrategy && firstTicker) {{
                    document.getElementById('strategy-select').value = firstStrategy;
                    document.getElementById('stock-select').value = firstTicker;
                    loadReport();
                }}
            }}
        }});
        
        function loadReport() {{
            const strategy = document.getElementById('strategy-select').value;
            const stock = document.getElementById('stock-select').value;
            const report = RESULTS[strategy]?.[stock];
            
            if (report && report.file) {{
                // 加载图表
                document.getElementById('chart-frame').src = report.file;
                
                // 更新统计数据
                updateStats(report.stats);
                
                showNotification('✅ 报告加载成功', 'success');
            }} else {{
                document.getElementById('chart-frame').src = 'about:blank';
                clearStats();
                showNotification('❌ 未找到回测报告', 'error');
            }}
        }}
        
        function updateStats(stats) {{
            const format = (value, isPercent = false) => {{
                if (value == null || value === '--') return '--';
                if (typeof value === 'number') {{
                    if (isPercent) return value.toFixed(2) + '%';
                    if (Math.abs(value) >= 1000) return value.toFixed(0);
                    return value.toFixed(2);
                }}
                return value;
            }};
            
            document.getElementById('stat-name').textContent = stats.标的名称 || '--';
            document.getElementById('stat-period').textContent = 
                `${{stats.数据起点 || '--'}} 至 ${{stats.数据终点 || '--'}}`;
            document.getElementById('stat-count').textContent = stats.数据条数 || '--';
            document.getElementById('stat-initial').textContent = format(stats.初始资金);
            document.getElementById('stat-final').textContent = format(stats['Equity Final [$]']);
            document.getElementById('stat-return').textContent = format(stats['Return [%]'], true);
            document.getElementById('stat-annual').textContent = format(stats['Return (Ann.) [%]'], true);
            document.getElementById('stat-drawdown').textContent = format(stats['Max. Drawdown [%]'], true);
            document.getElementById('stat-sharpe').textContent = format(stats['Sharpe Ratio']);
            document.getElementById('stat-trades').textContent = stats['# Trades'] || '--';
            document.getElementById('stat-winrate').textContent = format(stats['Win Rate [%]'], true);
            document.getElementById('stat-profit').textContent = format(stats['Profit Factor']);
            
            // 高亮关键指标
            highlightStats(stats);
        }}
        
        function highlightStats(stats) {{
            const highlight = (id, condition, goodColor = '#28a745', badColor = '#dc3545') => {{
                const el = document.getElementById(id);
                if (condition) {{
                    el.style.color = goodColor;
                    el.style.fontWeight = 'bold';
                }} else if (el) {{
                    el.style.color = '';
                    el.style.fontWeight = '';
                }}
            }};
            
            highlight('stat-sharpe', stats['Sharpe Ratio'] > 1);
            highlight('stat-drawdown', stats['Max. Drawdown [%]'] < -20, '#dc3545', '#28a745');
            highlight('stat-winrate', stats['Win Rate [%]'] > 60);
        }}
        
        function clearStats() {{
            const cells = document.querySelectorAll('#stats-table td:last-child');
            cells.forEach(cell => {{
                cell.textContent = '--';
                cell.style.color = '';
                cell.style.fontWeight = '';
            }});
        }}
        
        function downloadCSV() {{
            window.open('strategy_comparison.csv', '_blank');
            showNotification('📥 正在下载完整报告...', 'info');
        }}
        
        function showNotification(message, type) {{
            const notification = document.createElement('div');
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                z-index: 1000;
                animation: slideIn 0.3s ease;
            `;
            
            if (type === 'success') notification.style.background = '#28a745';
            else if (type === 'error') notification.style.background = '#dc3545';
            else notification.style.background = '#17a2b8';
            
            document.body.appendChild(notification);
            
            setTimeout(() => {{
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }}, 3000);
        }}
        
        // 添加CSS动画
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {{ from {{ transform: translateX(100%); opacity: 0; }} to {{ transform: translateX(0); opacity: 1; }} }}
            @keyframes slideOut {{ from {{ transform: translateX(0); opacity: 1; }} to {{ transform: translateX(100%); opacity: 0; }} }}
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>"""
    
    # 保存HTML文件
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 主页面已生成: {output_dir}/index.html")

# ------------------------------------------------------------------
# 7. 程序入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
