#!/usr/bin/env python3
# main.py - 多市场多策略回测系统

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 导入配置
# ------------------------------------------------------------------
try:
    # 尝试从config.py导入配置
    from config import STOCKS_CONFIG, BACKTEST_CONFIG, STRATEGY_PARAMS, ANALYSIS_CONFIG
except ImportError:
    # 如果config.py不存在，使用默认配置
    print("⚠️  config.py未找到，使用默认配置")
    
    STOCKS_CONFIG = {
        "港股": {
            "^HSI": "恒生指数",
            "0700.HK": "腾讯控股",
            "9988.HK": "阿里巴巴-SW",
            "3690.HK": "美团-W",
            "1810.HK": "小米集团-W",
        },
        "美股": {
            "SPY": "标普500 ETF",
            "QQQ": "纳指100 ETF",
            "AAPL": "苹果",
            "MSFT": "微软",
            "GOOGL": "谷歌",
            "AMZN": "亚马逊",
            "TSLA": "特斯拉",
            "NVDA": "英伟达",
        },
        "A股(美股ADR)": {
            "BABA": "阿里巴巴",
            "JD": "京东",
            "PDD": "拼多多",
            "NIO": "蔚来",
        }
    }
    
    BACKTEST_CONFIG = {
        "start_date": "2020-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "initial_cash": 100000,
        "commission": 0.002,
        "slippage": 0.001,
    }
    
    STRATEGY_PARAMS = {
        "RSI": {"upper": 70, "lower": 30, "window": 14},
        "SMA": {"fast": 20, "slow": 50},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BB": {"window": 20, "dev": 2.0},
        "Stoch": {"k_period": 14, "d_period": 3, "smooth_k": 3}
    }
    
    ANALYSIS_CONFIG = {
        "risk_free_rate": 0.02,
        "benchmark": "^GSPC",
        "max_drawdown_limit": 0.2,
        "min_sharpe_ratio": 1.0,
    }

# ------------------------------------------------------------------
# 1. 策略定义
# ------------------------------------------------------------------
class RsiOscillator(Strategy):
    Name = "RSI_Oscillator"
    
    def init(self):
        # 从配置获取参数
        params = STRATEGY_PARAMS.get("RSI", {"upper": 70, "lower": 30, "window": 14})
        self.upper = params["upper"]
        self.lower = params["lower"]
        self.window = params["window"]
        
        self.rsi = self.I(talib.RSI, self.data.Close, self.window)
        self.buy_signal = pd.Series(index=self.data.Close.index, dtype=bool)
        self.sell_signal = pd.Series(index=self.data.Close.index, dtype=bool)
    
    def next(self):
        if crossover(self.rsi, self.upper):
            self.position.close()
            self.sell_signal.iloc[-1] = True
        elif crossover(self.lower, self.rsi) and not self.position:
            self.buy()
            self.buy_signal.iloc[-1] = True

class SmaCrossover(Strategy):
    Name = "SMA_Crossover"
    
    def init(self):
        params = STRATEGY_PARAMS.get("SMA", {"fast": 20, "slow": 50})
        self.fast = params["fast"]
        self.slow = params["slow"]
        
        self.sma_f = self.I(talib.SMA, self.data.Close, self.fast)
        self.sma_s = self.I(talib.SMA, self.data.Close, self.slow)
        self.buy_signal = pd.Series(index=self.data.Close.index, dtype=bool)
        self.sell_signal = pd.Series(index=self.data.Close.index, dtype=bool)
    
    def next(self):
        if crossover(self.sma_f, self.sma_s):
            if not self.position:
                self.buy()
                self.buy_signal.iloc[-1] = True
        elif crossover(self.sma_s, self.sma_f) and self.position:
            self.position.close()
            self.sell_signal.iloc[-1] = True

class MacdCrossover(Strategy):
    Name = "MACD_Crossover"
    
    def init(self):
        params = STRATEGY_PARAMS.get("MACD", {"fast": 12, "slow": 26, "signal": 9})
        self.fast = params["fast"]
        self.slow = params["slow"]
        self.signal = params["signal"]
        
        macd, signal, hist = talib.MACD(
            self.data.Close, 
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal
        )
        self.macd_line = self.I(lambda: macd)
        self.signal_line = self.I(lambda: signal)
        self.histogram = self.I(lambda: hist)
        self.buy_signal = pd.Series(index=self.data.Close.index, dtype=bool)
        self.sell_signal = pd.Series(index=self.data.Close.index, dtype=bool)
    
    def next(self):
        if crossover(self.macd_line, self.signal_line):
            if not self.position:
                self.buy()
                self.buy_signal.iloc[-1] = True
        elif crossover(self.signal_line, self.macd_line) and self.position:
            self.position.close()
            self.sell_signal.iloc[-1] = True

class BollingerBandsStrategy(Strategy):
    Name = "Bollinger_Bands"
    
    def init(self):
        params = STRATEGY_PARAMS.get("BB", {"window": 20, "dev": 2.0})
        self.window = params["window"]
        self.dev = params["dev"]
        
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.window,
            nbdevup=self.dev, nbdevdn=self.dev, matype=0
        )
        self.buy_signal = pd.Series(index=self.data.Close.index, dtype=bool)
        self.sell_signal = pd.Series(index=self.data.Close.index, dtype=bool)
    
    def next(self):
        price = self.data.Close[-1]
        
        # 价格跌破下轨买入，突破上轨卖出
        if price < self.bb_lower[-1] and not self.position:
            self.buy()
            self.buy_signal.iloc[-1] = True
        elif price > self.bb_upper[-1] and self.position:
            self.position.close()
            self.sell_signal.iloc[-1] = True

class StochasticStrategy(Strategy):
    Name = "Stochastic_Oscillator"
    
    def init(self):
        params = STRATEGY_PARAMS.get("Stoch", {"k_period": 14, "d_period": 3, "smooth_k": 3})
        self.k_period = params["k_period"]
        self.d_period = params["d_period"]
        self.smooth_k = params["smooth_k"]
        
        slowk, slowd = talib.STOCH(
            self.data.High, self.data.Low, self.data.Close,
            fastk_period=self.k_period,
            slowk_period=self.smooth_k,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0
        )
        self.slowk = self.I(lambda: slowk)
        self.slowd = self.I(lambda: slowd)
        self.buy_signal = pd.Series(index=self.data.Close.index, dtype=bool)
        self.sell_signal = pd.Series(index=self.data.Close.index, dtype=bool)
    
    def next(self):
        # K线上穿D线且处于超卖区买入，下穿且处于超买区卖出
        if (crossover(self.slowk, self.slowd) and 
            self.slowk[-1] < 20 and not self.position):
            self.buy()
            self.buy_signal.iloc[-1] = True
        elif (crossover(self.slowd, self.slowk) and 
              self.slowk[-1] > 80 and self.position):
            self.position.close()
            self.sell_signal.iloc[-1] = True

# ------------------------------------------------------------------
# 2. 数据获取函数
# ------------------------------------------------------------------
def get_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """获取股票数据"""
    if start is None:
        start = BACKTEST_CONFIG["start_date"]
    if end is None:
        end = BACKTEST_CONFIG["end_date"]
    
    print(f"📥 正在获取 {ticker} 数据 ({start} 到 {end})...")
    
    try:
        # 使用yfinance下载数据
        df = yf.download(
            ticker, 
            start=start, 
            end=end,
            progress=False,
            auto_adjust=True
        )
        
        if df.empty:
            print(f"⚠️  {ticker}: 没有获取到数据")
            return pd.DataFrame()
        
        # 清理列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 确保必要的列存在
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  {ticker}: 缺少列 {missing_cols}")
            # 尝试修复：如果是港股，可能有不同的列名
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            else:
                return pd.DataFrame()
        
        # 处理时区
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        # 检查是否有足够的数据
        if len(df) < 20:
            print(f"⚠️  {ticker}: 数据太少 ({len(df)} 条)")
            return pd.DataFrame()
        
        # 添加收益数据
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        print(f"✅  {ticker}: 获取 {len(df)} 条数据 (最新: {df.index[-1].date()})")
        return df
        
    except Exception as e:
        print(f"❌  {ticker} 数据获取失败: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------
# 3. 性能分析函数
# ------------------------------------------------------------------
def calculate_additional_metrics(stats: dict, returns: pd.Series) -> dict:
    """计算额外的性能指标"""
    if returns.empty or len(returns) < 10:
        return {}
    
    try:
        # 计算最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # 计算风险调整收益
        excess_returns = returns - ANALYSIS_CONFIG["risk_free_rate"] / 252
        
        metrics = {
            # 风险指标
            "Max_Drawdown_Value": float(drawdown.min()) if not drawdown.empty else 0,
            "Volatility_Daily": float(returns.std()) if len(returns) > 1 else 0,
            "Volatility_Annual": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0,
            
            # 收益指标
            "Total_Return": float(cumulative.iloc[-1] - 1) if not cumulative.empty else 0,
            "Annualized_Return": float((1 + returns.mean()) ** 252 - 1) if len(returns) > 0 else 0,
            
            # 比率指标
            "Sortino_Ratio": float(excess_returns.mean() / returns[returns < 0].std() * np.sqrt(252)) 
                            if len(returns[returns < 0]) > 1 else 0,
            "Treynor_Ratio": float(excess_returns.mean() / returns.std()) if len(returns) > 1 else 0,
        }
        
        return metrics
    except Exception as e:
        print(f"⚠️  计算额外指标时出错: {e}")
        return {}

# ------------------------------------------------------------------
# 4. 主程序
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("📊 多市场多策略回测系统")
    print("=" * 60)
    
    # 合并所有股票标的
    ALL_TICKERS = {}
    for market, tickers in STOCKS_CONFIG.items():
        ALL_TICKERS.update(tickers)
    
    # 所有策略
    STRATEGIES = [
        RsiOscillator,
        SmaCrossover,
        MacdCrossover,
        BollingerBandsStrategy,
        StochasticStrategy,
    ]
    
    # 输出目录
    OUT_DIR = "public"
    REPORT_DIR = os.path.join(OUT_DIR, "reports")
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
    
    reports_map = {}
    all_reports = []
    
    total_tests = len(STRATEGIES) * len(ALL_TICKERS)
    completed_tests = 0
    
    print(f"🎯 总测试组合: {len(STRATEGIES)} 策略 × {len(ALL_TICKERS)} 标的 = {total_tests}")
    print(f"📅 回测期间: {BACKTEST_CONFIG['start_date']} 到 {BACKTEST_CONFIG['end_date']}")
    print()
    
    for Stg in STRATEGIES:
        stg_name = Stg.Name
        reports_map[stg_name] = {}
        print(f"\n📈 执行策略: {stg_name}")
        print("-" * 40)
        
        for ticker, desc in ALL_TICKERS.items():
            try:
                # 获取数据
                data = get_data(ticker)
                
                if data.empty or len(data) < 50:
                    print(f"   ⏭️  跳过 {desc} ({ticker}): 数据不足")
                    continue
                
                # 运行回测
                bt = Backtest(
                    data, 
                    Stg, 
                    cash=BACKTEST_CONFIG["initial_cash"],
                    commission=BACKTEST_CONFIG["commission"],
                    exclusive_orders=True
                )
                
                stats = bt.run()
                
                # 获取交易数据用于计算额外指标
                returns = data['Returns'].dropna() if 'Returns' in data.columns else pd.Series()
                extra_metrics = calculate_additional_metrics(stats, returns)
                
                # 生成报告文件名
                safe_ticker = ticker.replace("^", "").replace(".", "_").replace("-", "_")
                fname = f"{stg_name}_{safe_ticker}.html"
                report_path = os.path.join(REPORT_DIR, fname)
                
                # 保存图表
                try:
                    bt.plot(
                        filename=report_path,
                        open_browser=False,
                        superimpose=False,
                        plot_volume=False,
                        plot_drawdown=True
                    )
                except Exception as e:
                    print(f"   ⚠️  图表生成失败: {e}")
                    # 继续执行，图表不是必须的
                
                # 存储结果
                reports_map[stg_name][ticker] = f"reports/{fname}"
                
                # 存储统计数据
                stats_data = {
                    **extra_metrics,
                    "标的名称": desc,
                    "数据起点": str(data.index[0].date()) if not data.empty else "",
                    "数据终点": str(data.index[-1].date()) if not data.empty else "",
                    "数据条数": len(data),
                    "初始资金": BACKTEST_CONFIG["initial_cash"],
                    "手续费率": BACKTEST_CONFIG["commission"],
                }
                
                # 添加原始统计指标
                for key, value in stats.items():
                    if isinstance(value, (int, float, str, bool)):
                        stats_data[key] = value
                
                reports_map[stg_name][ticker + "_stats"] = stats_data
                
                # 收集报告数据
                all_reports.append({
                    "策略": stg_name,
                    "标的代码": ticker,
                    "标的名称": desc,
                    "年化收益%": stats_data.get("Return (Ann.) [%]", 0),
                    "夏普比率": stats_data.get("Sharpe Ratio", 0),
                    "最大回撤%": stats_data.get("Max. Drawdown [%]", 0),
                    "总收益率%": stats_data.get("Total_Return", 0) * 100,
                    "胜率%": stats_data.get("Win Rate [%]", 0),
                    "交易次数": stats_data.get("# Trades", 0),
                    "报告文件": f"reports/{fname}",
                })
                
                completed_tests += 1
                trades_count = stats_data.get("# Trades", 0)
                print(f"   ✅  {desc} ({ticker}): 完成 ({trades_count} 笔交易)")
                
            except Exception as e:
                print(f"   ❌  {desc} ({ticker}) 回测失败: {str(e)[:100]}")
                continue
    
    print(f"\n🎉 回测完成: {completed_tests}/{total_tests} 个组合")
    
    # 生成策略对比报告
    if all_reports:
        df_report = pd.DataFrame(all_reports)
        
        # 按夏普比率排序
        df_report = df_report.sort_values("夏普比率", ascending=False)
        
        # 保存为CSV
        csv_path = os.path.join(OUT_DIR, "strategy_comparison.csv")
        df_report.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"📊 策略对比报告已保存: {csv_path}")
        
        # 生成最佳策略推荐
        if not df_report.empty:
            best_by_sharpe = df_report.iloc[0]
            print(f"🏆 最佳夏普比率: {best_by_sharpe['策略']} + {best_by_sharpe['标的名称']}")
            print(f"   夏普比率: {best_by_sharpe['夏普比率']:.2f}, 年化收益: {best_by_sharpe['年化收益%']:.1f}%")
    
    # ------------------------------------------------------------------
    # 5. 生成增强的 index.html
    # ------------------------------------------------------------------
    # 构建下拉选项
    strategy_options = ""
    for strategy in STRATEGIES:
        strategy_options += f'<option value="{strategy.Name}">{strategy.Name.replace("_", " ")}</option>\n'
    
    ticker_options = ""
    for market, tickers in STOCKS_CONFIG.items():
        ticker_options += f'<optgroup label="{market}">\n'
        for ticker, name in tickers.items():
            ticker_options += f'  <option value="{ticker}">{name} ({ticker})</option>\n'
        ticker_options += '</optgroup>\n'
    
    reports_json = json.dumps(reports_map, ensure_ascii=False, indent=2)
    
    # 统计表格
    stats_table = """
    <table class="stats">
        <thead><tr><th class="left">指标</th><th>数值</th></tr></thead>
        <tbody id="stats-body">
            <tr><td class="left">数据起点</td><td id="st_数据起点">--</td></tr>
            <tr><td class="left">数据终点</td><td id="st_数据终点">--</td></tr>
            <tr><td class="left">数据条数</td><td id="st_数据条数">--</td></tr>
            <tr><td class="left">交易次数</td><td id="st_Trades">--</td></tr>
            <tr><td class="left">胜率%</td><td id="st_WinRate">--</td></tr>
            <tr><td class="left">年化收益%</td><td id="st_ReturnAnn">--</td></tr>
            <tr><td class="left">夏普比率</td><td id="st_SharpeRatio">--</td></tr>
            <tr><td class="left">最大回撤%</td><td id="st_MaxDrawdown">--</td></tr>
            <tr><td class="left">总收益率%</td><td id="st_TotalReturn">--</td></tr>
            <tr><td class="left">索提诺比率</td><td id="st_SortinoRatio">--</td></tr>
            <tr><td class="left">年化波动率%</td><td id="st_VolatilityAnnual">--</td></tr>
        </tbody>
    </table>
    """
    
    # 完整的HTML页面
    index_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多市场多策略回测系统</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .control-panel {{
            padding: 25px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}
        .input-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            margin-bottom: 20px;
        }}
        .input-box {{
            flex: 1;
            min-width: 250px;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }}
        select, input {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }}
        select:focus, input:focus {{
            border-color: #667eea;
            outline: none;
        }}
        .btn-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        button {{
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .content {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            padding: 25px;
            min-height: 600px;
        }}
        .report-container {{
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        .stats-panel {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            overflow-y: auto;
            max-height: 600px;
        }}
        .stats h3 {{
            margin-bottom: 20px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .stats table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats th {{
            background: #e9ecef;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }}
        .stats td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .stats tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
            background: #f8f9fa;
        }}
        @media (max-width: 1024px) {{
            .content {{ grid-template-columns: 1fr; }}
            .stats-panel {{ max-height: 400px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> 多市场多策略回测系统</h1>
            <p>覆盖美股、港股、A股ADR，支持多种技术指标策略</p>
        </div>
        
        <div class="control-panel">
            <div class="input-group">
                <div class="input-box">
                    <label><i class="fas fa-chart-bar"></i> 选择策略</label>
                    <select id="strategy-select">
                        {strategy_options}
                    </select>
                </div>
                
                <div class="input-box">
                    <label><i class="fas fa-dollar-sign"></i> 选择标的</label>
                    <select id="ticker-select">
                        {ticker_options}
                    </select>
                </div>
            </div>
            
            <div class="btn-group">
                <button onclick="loadReport()" class="btn-primary">
                    <i class="fas fa-play"></i> 加载报告
                </button>
                <button onclick="downloadReport()" class="btn-primary">
                    <i class="fas fa-download"></i> 下载数据
                </button>
            </div>
        </div>
        
        <div class="content">
            <div class="report-container">
                <iframe id="report-iframe" src=""></iframe>
            </div>
            
            <div class="stats-panel">
                <h3><i class="fas fa-chart-pie"></i> 性能指标</h3>
                {stats_table}
            </div>
        </div>
        
        <div class="footer">
            <p>数据来源: Yahoo Finance | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>风险提示: 回测结果不代表未来表现，投资有风险，入市需谨慎</p>
        </div>
    </div>
    
    <script>
        const REPORTS_MAP = {reports_json};
        
        window.onload = function() {{
            // 默认加载第一个策略和标的
            if (REPORTS_MAP && Object.keys(REPORTS_MAP).length > 0) {{
                const firstStrategy = Object.keys(REPORTS_MAP)[0];
                const firstTicker = Object.keys(REPORTS_MAP[firstStrategy]).find(key => !key.includes('_stats'));
                
                if (firstStrategy && firstTicker) {{
                    document.getElementById('strategy-select').value = firstStrategy;
                    document.getElementById('ticker-select').value = firstTicker;
                    loadSpecificReport(firstStrategy, firstTicker);
                }}
            }}
        }};
        
        function loadReport() {{
            const strategy = document.getElementById('strategy-select').value;
            const ticker = document.getElementById('ticker-select').value;
            loadSpecificReport(strategy, ticker);
        }}
        
        function loadSpecificReport(strategy, ticker) {{
            const filename = REPORTS_MAP[strategy]?.[ticker];
            const iframe = document.getElementById('report-iframe');
            
            if (filename) {{
                iframe.src = filename;
                updateStats(REPORTS_MAP[strategy][ticker + '_stats'] || {{}});
                showNotification(`加载成功: ${{strategy}} - ${{ticker}}`, 'success');
            }} else {{
                iframe.src = "about:blank";
                showNotification(`找不到 ${{strategy}} 策略与 ${{ticker}} 标的的报告`, 'error');
                clearStats();
            }}
        }}
        
        function updateStats(stats) {{
            const formatValue = (value, type = 'number') => {{
                if (value === null || value === undefined || value === '--') return '--';
                if (type === 'percent') return (value * 100).toFixed(2) + '%';
                if (type === 'number') return typeof value === 'number' ? value.toFixed(2) : value;
                return value;
            }};
            
            // 更新统计值
            const mappings = {{
                'st_数据起点': ['数据起点', ''],
                'st_数据终点': ['数据终点', ''],
                'st_数据条数': ['数据条数', ''],
                'st_Trades': ['# Trades', ''],
                'st_WinRate': ['Win Rate [%]', 'percent'],
                'st_ReturnAnn': ['Return (Ann.) [%]', 'percent'],
                'st_SharpeRatio': ['Sharpe Ratio', 'number'],
                'st_MaxDrawdown': ['Max. Drawdown [%]', 'percent'],
                'st_TotalReturn': ['Total_Return', 'percent'],
                'st_SortinoRatio': ['Sortino_Ratio', 'number'],
                'st_VolatilityAnnual': ['Volatility_Annual', 'percent'],
            }};
            
            for (const [id, [key, type]] of Object.entries(mappings)) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.textContent = formatValue(stats[key], type);
                    
                    // 高亮好的指标
                    if (key === 'Sharpe Ratio' && stats[key] > 1) {{
                        element.style.color = '#28a745';
                        element.style.fontWeight = 'bold';
                    }} else if (key === 'Max. Drawdown [%]' && stats[key] < -0.1) {{
                        element.style.color = '#dc3545';
                        element.style.fontWeight = 'bold';
                    }} else {{
                        element.style.color = '';
                        element.style.fontWeight = '';
                    }}
                }}
            }}
        }}
        
        function clearStats() {{
            const statsElements = document.querySelectorAll('[id^="st_"]');
            statsElements.forEach(el => {{
                el.textContent = '--';
                el.style.color = '';
                el.style.fontWeight = '';
            }});
        }}
        
        function downloadReport() {{
            const strategy = document.getElementById('strategy-select').value;
            const ticker = document.getElementById('ticker-select').value;
            const filename = REPORTS_MAP[strategy]?.[ticker];
            
            if (filename) {{
                // 下载CSV数据
                window.open('strategy_comparison.csv', '_blank');
                showNotification('正在下载数据...', 'success');
            }} else {{
                showNotification('没有可下载的数据', 'error');
            }}
        }}
        
        function showNotification(message, type) {{
            // 简单的通知实现
            const notification = document.createElement('div');
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 25px;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                z-index: 1000;
                animation: slideIn 0.3s ease;
            `;
            
            if (type === 'success') {{
                notification.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
            }} else if (type === 'error') {{
                notification.style.background = 'linear-gradient(135deg, #dc3545 0%, #fd7e14 100%)';
            }}
            
            document.body.appendChild(notification);
            
            setTimeout(() => {{
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }}, 3000);
        }}
        
        // 添加CSS动画
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {{
                from {{ transform: translateX(100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            @keyframes slideOut {{
                from {{ transform: translateX(0); opacity: 1; }}
                to {{ transform: translateX(100%); opacity: 0; }}
            }}
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>"""
    
    # 保存HTML文件
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    
    print(f"✅ 主页面已生成: {os.path.join(OUT_DIR, 'index.html')}")
    print(f"📁 报告总数: {completed_tests}")
    
    # 生成市场数据汇总
    print("\n📋 市场数据汇总:")
    print("-" * 40)
    for market, tickers in STOCKS_CONFIG.items():
        print(f"{market}: {len(tickers)} 个标的")
    
    print("=" * 60)
    
    return completed_tests

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
