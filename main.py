[file name]: requirements.txt
[file content begin]
yfinance
pandas
numpy
backtesting
bokeh==3.2.1
TA-Lib
plotly
matplotlib
seaborn
scipy
[file content end]

[file name]: config.py
[file content begin]
#!/usr/bin/env python3
# config.py - 配置文件

# 股票配置
STOCKS_CONFIG = {
    "港股": {
        "^HSI": "恒生指数",
        "0700.HK": "腾讯控股",
        "9988.HK": "阿里巴巴-SW",
        "3690.HK": "美团-W",
        "1810.HK": "小米集团-W",
        "1211.HK": "比亚迪股份",
        "0005.HK": "汇丰控股",
        "1299.HK": "友邦保险",
        "0941.HK": "中国移动",
        "0388.HK": "香港交易所",
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
        "META": "Meta",
        "BRK-B": "伯克希尔",
        "JPM": "摩根大通",
        "V": "Visa",
        "JNJ": "强生",
        "WMT": "沃尔玛",
        "MA": "万事达卡",
    },
    "A股(美股ADR)": {
        "BABA": "阿里巴巴",
        "JD": "京东",
        "PDD": "拼多多",
        "BIDU": "百度",
        "NIO": "蔚来",
        "LI": "理想汽车",
        "XPEV": "小鹏汽车",
        "TCEHY": "腾讯(OTC)",
    }
}

# 回测参数
BACKTEST_CONFIG = {
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "initial_cash": 100000,
    "commission": 0.002,  # 0.2% 手续费
    "slippage": 0.001,    # 0.1% 滑点
}

# 策略参数
STRATEGY_PARAMS = {
    "RSI": {
        "upper": 70,
        "lower": 30,
        "window": 14
    },
    "SMA": {
        "fast": 20,
        "slow": 50
    },
    "MACD": {
        "fast": 12,
        "slow": 26,
        "signal": 9
    },
    "BB": {
        "window": 20,
        "dev": 2.0
    },
    "Stoch": {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3
    }
}

# 分析配置
ANALYSIS_CONFIG = {
    "risk_free_rate": 0.02,  # 无风险利率
    "benchmark": "^GSPC",    # 标普500作为基准
    "max_drawdown_limit": 0.2,  # 最大回撤限制
    "min_sharpe_ratio": 1.0,    # 最低夏普比率
}
[file content end]

[file name]: main.py
[file content begin]
#!/usr/bin/env python3
# main.py
import os
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

from config import STOCKS_CONFIG, BACKTEST_CONFIG, STRATEGY_PARAMS, ANALYSIS_CONFIG

# ------------------------------------------------------------------
# 1. 扩展策略定义
# ------------------------------------------------------------------
class RsiOscillator(Strategy):
    Name = "RSI_Oscillator"
    upper = STRATEGY_PARAMS["RSI"]["upper"]
    lower = STRATEGY_PARAMS["RSI"]["lower"]
    window = STRATEGY_PARAMS["RSI"]["window"]
    
    def init(self):
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
    fast = STRATEGY_PARAMS["SMA"]["fast"]
    slow = STRATEGY_PARAMS["SMA"]["slow"]
    
    def init(self):
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
    fast = STRATEGY_PARAMS["MACD"]["fast"]
    slow = STRATEGY_PARAMS["MACD"]["slow"]
    signal = STRATEGY_PARAMS["MACD"]["signal"]
    
    def init(self):
        macd, signal, hist = talib.MACD(self.data.Close, 
                                       fastperiod=self.fast,
                                       slowperiod=self.slow,
                                       signalperiod=self.signal)
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
    window = STRATEGY_PARAMS["BB"]["window"]
    dev = STRATEGY_PARAMS["BB"]["dev"]
    
    def init(self):
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
    k_period = STRATEGY_PARAMS["Stoch"]["k_period"]
    d_period = STRATEGY_PARAMS["Stoch"]["d_period"]
    smooth_k = STRATEGY_PARAMS["Stoch"]["smooth_k"]
    
    def init(self):
        slowk, slowd = talib.STOCH(self.data.High, self.data.Low, self.data.Close,
                                  fastk_period=self.k_period,
                                  slowk_period=self.smooth_k,
                                  slowk_matype=0,
                                  slowd_period=self.d_period,
                                  slowd_matype=0)
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

class DualMovingAverageStrategy(Strategy):
    """双均线策略 + 成交量过滤"""
    Name = "Dual_MA_Volume"
    
    def init(self):
        self.sma_short = self.I(talib.SMA, self.data.Close, 10)
        self.sma_long = self.I(talib.SMA, self.data.Close, 30)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.buy_signal = pd.Series(index=self.data.Close.index, dtype=bool)
        self.sell_signal = pd.Series(index=self.data.Close.index, dtype=bool)
    
    def next(self):
        # 成交量高于平均才交易
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1] * 1.2
        
        if (crossover(self.sma_short, self.sma_long) and 
            volume_ok and not self.position):
            self.buy()
            self.buy_signal.iloc[-1] = True
        elif (crossover(self.sma_long, self.sma_short) and 
              self.position):
            self.position.close()
            self.sell_signal.iloc[-1] = True

# ------------------------------------------------------------------
# 2. 增强的数据获取函数
# ------------------------------------------------------------------
def get_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """获取股票数据，支持港股、美股和指数"""
    if start is None:
        start = BACKTEST_CONFIG["start_date"]
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📥 正在获取 {ticker} 数据 ({start} 到 {end})...")
    
    try:
        # yfinance 自动处理港股后缀
        stock = yf.Ticker(ticker)
        
        # 获取历史数据
        df = stock.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            print(f"⚠️  {ticker}: 没有获取到数据")
            return pd.DataFrame()
        
        # 清理列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 确保必要的列存在
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️  {ticker}: 缺少 {col} 列")
                return pd.DataFrame()
        
        # 处理时区
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        # 添加技术指标需要的数据
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        print(f"✅  {ticker}: 获取 {len(df)} 条数据 (最新: {df.index[-1].date()})")
        return df
        
    except Exception as e:
        print(f"❌  {ticker} 数据获取失败: {e}")
        return pd.DataFrame()

def get_market_data(ticker: str) -> dict:
    """获取市场数据信息"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        market_data = {
            "symbol": ticker,
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "quoteType": info.get("quoteType", "N/A"),
        }
        return market_data
    except:
        return {"symbol": ticker, "name": ticker}

# ------------------------------------------------------------------
# 3. 性能分析函数
# ------------------------------------------------------------------
def calculate_additional_metrics(stats: dict, returns: pd.Series) -> dict:
    """计算额外的性能指标"""
    if returns.empty:
        return {}
    
    # 计算最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # 计算风险调整收益
    excess_returns = returns - ANALYSIS_CONFIG["risk_free_rate"] / 252
    
    metrics = {
        # 风险指标
        "Max_Drawdown_Value": float(drawdown.min()),
        "Volatility_Daily": float(returns.std()),
        "Volatility_Annual": float(returns.std() * np.sqrt(252)),
        "VaR_95": float(returns.quantile(0.05)),
        "CVaR_95": float(returns[returns <= returns.quantile(0.05)].mean()),
        
        # 收益指标
        "Total_Return": float(cumulative.iloc[-1] - 1),
        "Annualized_Return": float((1 + returns.mean()) ** 252 - 1),
        "Excess_Return": float(excess_returns.mean() * 252),
        
        # 比率指标
        "Sortino_Ratio": float(excess_returns.mean() / returns[returns < 0].std()) * np.sqrt(252),
        "Treynor_Ratio": float(excess_returns.mean() / returns.std()),
        "Information_Ratio": float(excess_returns.mean() / excess_returns.std()),
        
        # 交易质量指标
        "Profit_Loss_Ratio": abs(stats.get('Avg. Trade [%]', 0) / 
                               (stats.get('Worst Trade [%]', -1) if stats.get('Worst Trade [%]', 0) < 0 else -1)),
        "Recovery_Factor": abs(stats.get('Equity Final [$]', 0) - BACKTEST_CONFIG["initial_cash"]) / 
                          abs(stats.get('Equity Peak [$]', BACKTEST_CONFIG["initial_cash"]) * 
                          stats.get('Max. Drawdown [%]', 1) / 100),
    }
    
    # 合并原始统计
    for key, value in stats.items():
        if isinstance(value, (int, float, str)):
            metrics[key] = value
    
    return metrics

def generate_strategy_report(stats_dict: dict) -> pd.DataFrame:
    """生成策略对比报告"""
    report_data = []
    
    for strategy, tickers in stats_dict.items():
        for ticker, stats in tickers.items():
            if "_stats" in ticker:
                continue
            
            stat_key = ticker + "_stats"
            if stat_key in stats_dict[strategy]:
                metrics = stats_dict[strategy][stat_key]
                report_data.append({
                    "策略": strategy,
                    "标的": ticker,
                    "年化收益%": metrics.get("Return (Ann.) [%]", 0),
                    "夏普比率": metrics.get("Sharpe Ratio", 0),
                    "最大回撤%": metrics.get("Max. Drawdown [%]", 0),
                    "胜率%": metrics.get("Win Rate [%]", 0),
                    "交易次数": metrics.get("# Trades", 0),
                    "盈利因子": metrics.get("Profit Factor", 0),
                    "索提诺比率": metrics.get("Sortino_Ratio", 0),
                    "总收益率%": metrics.get("Total_Return", 0) * 100,
                })
    
    return pd.DataFrame(report_data)

# ------------------------------------------------------------------
# 4. 主程序
# ------------------------------------------------------------------
if __name__ == "__main__":
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
        DualMovingAverageStrategy,
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
            # 获取数据
            data = get_data(ticker, 
                          BACKTEST_CONFIG["start_date"], 
                          BACKTEST_CONFIG["end_date"])
            
            if data.empty or len(data) < 100:
                print(f"   ⏭️  跳过 {desc} ({ticker}): 数据不足")
                continue
            
            # 确定所需的最小数据长度
            strategy_params = STRATEGY_PARAMS.get(stg_name.split('_')[0], {})
            min_data_needed = max(strategy_params.values()) if strategy_params else 50
            min_data_needed = max(min_data_needed, 100)  # 至少100条数据
            
            if len(data) < min_data_needed:
                print(f"   ⏭️  跳过 {desc} ({ticker}): 数据长度不足 ({len(data)} < {min_data_needed})")
                continue
            
            try:
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
                trades = stats['_trades'] if '_trades' in stats else pd.DataFrame()
                returns = data['Returns'].dropna()
                
                # 计算额外指标
                extra_metrics = calculate_additional_metrics(stats, returns)
                
                # 生成报告文件名
                safe_ticker = ticker.replace("^", "").replace(".", "_").replace("-", "_")
                fname = f"{stg_name}_{safe_ticker}.html"
                report_path = os.path.join(REPORT_DIR, fname)
                
                # 保存图表
                bt.plot(
                    filename=report_path,
                    open_browser=False,
                    superimpose=False,
                    plot_width=1200,
                    plot_equity=True,
                    plot_return=True,
                    plot_pl=True,
                    plot_volume=True,
                    plot_drawdown=True
                )
                
                # 存储结果
                reports_map[stg_name][ticker] = f"reports/{fname}"
                
                # 存储统计数据
                stats_data = {
                    **extra_metrics,
                    "标的名称": desc,
                    "数据起点": str(data.index[0].date()),
                    "数据终点": str(data.index[-1].date()),
                    "数据条数": len(data),
                    "初始资金": BACKTEST_CONFIG["initial_cash"],
                    "手续费率": BACKTEST_CONFIG["commission"],
                }
                
                reports_map[stg_name][ticker + "_stats"] = stats_data
                
                # 收集报告数据
                all_reports.append({
                    "策略": stg_name,
                    "标的代码": ticker,
                    "标的名称": desc,
                    "年化收益%": extra_metrics.get("Return (Ann.) [%]", 0),
                    "夏普比率": extra_metrics.get("Sharpe Ratio", 0),
                    "最大回撤%": extra_metrics.get("Max. Drawdown [%]", 0),
                    "总收益率%": extra_metrics.get("Total_Return", 0) * 100,
                    "胜率%": extra_metrics.get("Win Rate [%]", 0),
                    "交易次数": extra_metrics.get("# Trades", 0),
                    "报告文件": f"reports/{fname}",
                })
                
                completed_tests += 1
                print(f"   ✅  {desc} ({ticker}): 完成 ({len(trades)} 笔交易)")
                
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
        
        # 保存为HTML表格
        html_table = df_report.to_html(index=False, classes='report-table', border=0)
        
        print(f"📊 策略对比报告已保存: {csv_path}")
        
        # 生成最佳策略推荐
        best_by_sharpe = df_report.iloc[0]
        best_by_return = df_report.loc[df_report['年化收益%'].idxmax()]
        best_by_drawdown = df_report.loc[df_report['最大回撤%'].idxmin()]
        
        recommendations = f"""
        <div class="recommendations">
            <h3>🏆 最佳策略推荐</h3>
            <div class="rec-card">
                <h4>最佳夏普比率</h4>
                <p><strong>{best_by_sharpe['策略']}</strong> + {best_by_sharpe['标的名称']}</p>
                <p>夏普比率: {best_by_sharpe['夏普比率']:.2f}, 年化收益: {best_by_sharpe['年化收益%']:.1f}%</p>
            </div>
            <div class="rec-card">
                <h4>最高年化收益</h4>
                <p><strong>{best_by_return['策略']}</strong> + {best_by_return['标的名称']}</p>
                <p>年化收益: {best_by_return['年化收益%']:.1f}%, 最大回撤: {best_by_return['最大回撤%']:.1f}%</p>
            </div>
            <div class="rec-card">
                <h4>最低回撤</h4>
                <p><strong>{best_by_drawdown['策略']}</strong> + {best_by_drawdown['标的名称']}</p>
                <p>最大回撤: {best_by_drawdown['最大回撤%']:.1f}%, 夏普比率: {best_by_drawdown['夏普比率']:.2f}</p>
            </div>
        </div>
        """
    else:
        html_table = "<p>没有生成有效的回测报告</p>"
        recommendations = ""
    
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
    
    # 统计表格列
    stats_columns = [
        ("Start", "数据起点"),
        ("End", "数据终点"),
        ("数据条数", "数据条数"),
        ("# Trades", "交易次数"),
        ("Win Rate [%]", "胜率%"),
        ("Return (Ann.) [%]", "年化收益%"),
        ("Sharpe Ratio", "夏普比率"),
        ("Max. Drawdown [%]", "最大回撤%"),
        ("Sortino_Ratio", "索提诺比率"),
        ("Volatility_Annual", "年化波动率%"),
        ("Profit Factor", "盈利因子"),
        ("Total_Return", "总收益率%"),
        ("VaR_95", "VaR (95%)"),
        ("Recovery_Factor", "恢复因子"),
    ]
    
    stats_table = '<table class="stats">\n<thead><tr><th class="left">指标</th><th>数值</th></tr></thead>\n<tbody id="stats-body">\n'
    for key, name in stats_columns:
        html_id = key.replace(" ", "_").replace(".", "").replace("[", "").replace("]", "").replace("%", "")
        stats_table += f'  <tr><td class="left">{name}</td><td id="st_{html_id}">--</td></tr>\n'
    stats_table += '</tbody>\n</table>'
    
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
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        .btn-secondary:hover {{ background: #5a6268; }}
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
        .recommendations {{
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            color: white;
        }}
        .recommendations h3 {{
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        .rec-card {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }}
        .rec-card h4 {{
            margin-bottom: 10px;
            color: white;
        }}
        .rec-card p {{
            margin: 5px 0;
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
                
                <div class="input-box">
                    <label><i class="fas fa-search"></i> 快速查询</label>
                    <div style="display: flex; gap: 10px;">
                        <input id="symbol-input" type="text" placeholder="输入股票代码 (如: AAPL, 0700.HK)">
                        <button onclick="fetchSymbol()" class="btn-primary">
                            <i class="fas fa-search"></i> 查询
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="btn-group">
                <button onclick="loadReport()" class="btn-primary">
                    <i class="fas fa-play"></i> 运行回测
                </button>
                <button onclick="downloadReport()" class="btn-secondary">
                    <i class="fas fa-download"></i> 下载报告
                </button>
                <button onclick="showAllReports()" class="btn-secondary">
                    <i class="fas fa-list"></i> 查看全部
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
                {recommendations}
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
            const defaultStrategy = document.getElementById('strategy-select').value;
            const defaultTicker = document.getElementById('ticker-select').value;
            loadSpecificReport(defaultStrategy, defaultTicker);
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
                showNotification(`正在加载: ${{strategy}} - ${{ticker}}`, 'success');
            }} else {{
                iframe.src = "about:blank";
                showNotification(`找不到 ${{strategy}} 策略与 ${{ticker}} 标的的报告`, 'error');
                clearStats();
            }}
        }}
        
        function fetchSymbol() {{
            const symbol = document.getElementById('symbol-input').value.trim().toUpperCase();
            if (!symbol) {{
                showNotification('请输入股票代码', 'warning');
                return;
            }}
            
            // 在所有标的中查找
            const select = document.getElementById('ticker-select');
            for (let option of select.options) {{
                if (option.value === symbol) {{
                    select.value = symbol;
                    loadReport();
                    return;
                }}
            }}
            
            showNotification(`未找到 ${{symbol}}，请确认代码是否正确`, 'error');
        }}
        
        function updateStats(stats) {{
            const formatValue = (value, type = 'number') => {{
                if (value === null || value === undefined) return '--';
                if (type === 'percent') return (value * 100).toFixed(2) + '%';
                if (type === 'number') return typeof value === 'number' ? value.toFixed(2) : value;
                return value;
            }};
            
            // 更新所有统计值
            const mappings = {{
                'st_Start': ['数据起点', ''],
                'st_End': ['数据终点', ''],
                'st_数据条数': ['数据条数', ''],
                'st_Trades': ['交易次数', ''],
                'st_WinRate': ['胜率%', 'percent'],
                'st_ReturnAnn': ['年化收益%', 'percent'],
                'st_SharpeRatio': ['夏普比率', 'number'],
                'st_MaxDrawdown': ['最大回撤%', 'percent'],
                'st_SortinoRatio': ['索提诺比率', 'number'],
                'st_VolatilityAnnual': ['年化波动率%', 'percent'],
                'st_ProfitFactor': ['盈利因子', 'number'],
                'st_TotalReturn': ['总收益率%', 'percent'],
                'st_VaR95': ['VaR (95%)', 'percent'],
                'st_RecoveryFactor': ['恢复因子', 'number'],
            }};
            
            for (const [id, [key, type]] of Object.entries(mappings)) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.textContent = formatValue(stats[key], type);
                    
                    // 高亮好的指标
                    if (key === '夏普比率' && stats[key] > 1) {{
                        element.style.color = '#28a745';
                        element.style.fontWeight = 'bold';
                    }} else if (key === '最大回撤%' && stats[key] < -0.1) {{
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
                const link = document.createElement('a');
                link.href = filename;
                link.download = `${{strategy}}_${{ticker}}_report.html`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                showNotification('报告下载开始', 'success');
            }} else {{
                showNotification('没有可下载的报告', 'error');
            }}
        }}
        
        function showAllReports() {{
            // 在新窗口打开CSV报告
            window.open('strategy_comparison.csv', '_blank');
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
            }} else if (type === 'warning') {{
                notification.style.background = 'linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)';
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
    print("=" * 60)
    
    # 生成市场数据汇总
    print("\n📋 市场数据汇总:")
    print("-" * 40)
    for market, tickers in STOCKS_CONFIG.items():
        print(f"{market}: {len(tickers)} 个标的")
[file content end]

[file name]: analyze.py
[file content begin]
#!/usr/bin/env python3
# analyze.py - 数据分析工具
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_market_correlation():
    """分析市场相关性"""
    import yfinance as yf
    
    # 主要市场指数
    indices = {
        '^GSPC': '标普500',
        '^DJI': '道琼斯',
        '^IXIC': '纳斯达克',
        '^HSI': '恒生指数',
        '000001.SS': '上证指数',
        '^N225': '日经225',
        '^FTSE': '富时100',
    }
    
    print("📈 分析全球市场相关性...")
    
    # 获取数据
    data = {}
    for ticker, name in indices.items():
        try:
            df = yf.download(ticker, start='2020-01-01', progress=False)['Close']
            data[name] = df.pct_change().dropna()
            print(f"✅ 获取 {name} 数据: {len(df)} 天")
        except:
            print(f"❌ 无法获取 {name} 数据")
    
    # 创建相关性矩阵
    returns_df = pd.DataFrame(data)
    corr_matrix = returns_df.corr()
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .8})
    plt.title('全球主要指数收益率相关性矩阵 (2020-至今)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('public/market_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 相关性分析图已保存: public/market_correlation.png")
    
    return corr_matrix

def analyze_sector_performance():
    """分析行业表现"""
    sectors = {
        '科技': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
        '金融': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        '医疗': ['JNJ', 'PFE', 'UNH', 'ABT', 'MRK'],
        '消费': ['WMT', 'PG', 'KO', 'PEP', 'MCD'],
        '能源': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    }
    
    print("\n🏢 分析行业表现...")
    
    results = []
    for sector, stocks in sectors.items():
        sector_returns = []
        for stock in stocks:
            try:
                df = yf.download(stock, start='2020-01-01', progress=False)
                if not df.empty:
                    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                    sector_returns.append(total_return)
            except:
                continue
        
        if sector_returns:
            avg_return = np.mean(sector_returns)
            results.append({
                '行业': sector,
                '股票数量': len(sector_returns),
                '平均收益率%': avg_return,
                '最佳股票%': max(sector_returns) if sector_returns else 0,
                '最差股票%': min(sector_returns) if sector_returns else 0,
            })
    
    sector_df = pd.DataFrame(results)
    sector_df = sector_df.sort_values('平均收益率%', ascending=False)
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sector_df['行业'], sector_df['平均收益率%'], 
                   color=plt.cm.Set3(range(len(sector_df))))
    plt.xlabel('行业')
    plt.ylabel('平均收益率 (%)')
    plt.title('各行业平均收益率 (2020-至今)', fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('public/sector_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 行业表现分析图已保存: public/sector_performance.png")
    return sector_df

def generate_summary_report():
    """生成总结报告"""
    print("\n📊 生成总结报告...")
    
    try:
        # 读取回测结果
        df = pd.read_csv('public/strategy_comparison.csv')
        
        # 策略表现总结
        strategy_summary = df.groupby('策略').agg({
            '年化收益%': 'mean',
            '夏普比率': 'mean',
            '最大回撤%': 'mean',
            '胜率%': 'mean',
            '交易次数': 'mean'
        }).round(2)
        
        # 标的物表现总结
        ticker_summary = df.groupby('标的名称').agg({
            '年化收益%': 'mean',
            '夏普比率': 'mean',
            '最大回撤%': 'mean'
        }).round(2)
        
        # 生成HTML报告
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>回测总结报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .bad {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>📈 回测总结报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>🏆 最佳表现策略</h2>
                {df.nlargest(5, '夏普比率')[['策略', '标的名称', '夏普比率', '年化收益%', '最大回撤%']].to_html(index=False)}
            </div>
            
            <div class="section">
                <h2>📊 策略平均表现</h2>
                {strategy_summary.to_html()}
            </div>
            
            <div class="section">
                <h2>💹 标的物平均表现</h2>
                {ticker_summary.nlargest(10, '年化收益%').to_html()}
            </div>
            
            <div class="section">
                <h2>📋 完整结果</h2>
                <p>共 {len(df)} 个回测组合</p>
                {df.to_html(index=False)}
            </div>
        </body>
        </html>
        """
        
        with open('public/summary_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ 总结报告已生成: public/summary_report.html")
        print(f"📈 最佳策略: {df.loc[df['夏普比率'].idxmax(), '策略']} "
              f"({df.loc[df['夏普比率'].idxmax(), '标的名称']})")
        print(f"💰 最高收益: {df['年化收益%'].max():.1f}%")
        
    except Exception as e:
        print(f"❌ 生成总结报告失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("📊 数据分析工具")
    print("=" * 60)
    
    # 分析市场相关性
    corr_matrix = analyze_market_correlation()
    
    # 分析行业表现
    sector_df = analyze_sector_performance()
    
    # 生成总结报告
    generate_summary_report()
    
    print("\n" + "=" * 60)
    print("🎉 分析完成!")
    print("=" * 60)
[file content end]

[file name]: update_backtest.yml
[file content begin]
name: Daily Backtest and Analysis

on:
  push:
    branches: ["main"]
  schedule:
    - cron: '0 22 * * *'  # 每天 UTC 22:00 (台湾时间早上 06:00)
  workflow_dispatch:  # 允许手动触发

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ["3.9"]
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    # --- 安装系统依赖 ---
    - name: Install System Dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y wget build-essential
        
        # 安装 TA-Lib
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        cd ..
    
    # --- 安装 Python 依赖 ---
    - name: Install Python Dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements.txt
    
    # --- 运行回测 ---
    - name: Run Backtest Script
      run: |
        echo "开始运行回测..."
        python main.py
    
    # --- 运行数据分析 ---
    - name: Run Analysis Script
      run: |
        echo "开始数据分析..."
        python analyze.py
    
    # --- 部署到 GitHub Pages ---
    - name: Setup Pages
      uses: actions/configure-pages@v3
    
    - name: Upload artifact
      uses: actions/upload-pages-artifact@v2
      with:
        path: './public'
    
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v2
    
    # --- 生成运行状态报告 ---
    - name: Generate Status Report
      run: |
        echo "### 🚀 回测系统运行报告" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "**运行时间:** $(date '+%Y-%m-%d %H:%M:%S')" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "**环境信息:**" >> $GITHUB_STEP_SUMMARY
        echo "- Python: ${{ matrix.python-version }}" >> $GITHUB_STEP_SUMMARY
        echo "- 系统: Ubuntu Latest" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        
        # 检查生成的报告文件
        if [ -f "public/strategy_comparison.csv" ]; then
          echo "✅ **回测完成:**" >> $GITHUB_STEP_SUMMARY
          echo "策略报告已成功生成" >> $GITHUB_STEP_SUMMARY
        else
          echo "❌ **回测失败:**" >> $GITHUB_STEP_SUMMARY
          echo "未找到策略报告文件" >> $GITHUB_STEP_SUMMARY
        fi
        
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "**部署状态:** ${{ steps.deployment.outcome }}" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "📊 访问地址: https://${{ github.repository_owner }}.github.io/${{ github.event.repository.name }}/" >> $GITHUB_STEP_SUMMARY
[file content end]
