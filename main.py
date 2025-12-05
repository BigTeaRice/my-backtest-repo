#!/usr/bin/env python3
# main.py - 修复图表显示问题的多策略回测系统

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. 配置参数
# ------------------------------------------------------------------
CONFIG = {
    # 股票标的（简化为几个常用标的）
    "STOCKS": {
        "^HSI": "恒生指数",
        "0700.HK": "腾讯控股", 
        "9988.HK": "阿里巴巴",
        "AAPL": "苹果",
        "MSFT": "微软",
        "TSLA": "特斯拉",
    },
    
    # 回测参数
    "BACKTEST": {
        "start_date": "2023-01-01",  # 缩短时间，减少数据量
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
    return macd_line, signal_line

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """计算布林带"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """计算KDJ指标"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # 避免除零错误
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, 1)
    
    k_value = 100 * ((close - lowest_low) / denominator)
    d_value = k_value.rolling(window=d_period).mean()
    
    return k_value.fillna(50), d_value.fillna(50)

# ------------------------------------------------------------------
# 3. 策略定义（简化版）
# ------------------------------------------------------------------
class SmaStrategy(Strategy):
    """SMA双均线策略"""
    Name = "SMA策略"
    
    def init(self):
        self.sma_fast = self.I(calculate_sma, self.data.Close, 10)
        self.sma_slow = self.I(calculate_sma, self.data.Close, 30)
    
    def next(self):
        if crossover(self.sma_fast, self.sma_slow) and not self.position:
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast) and self.position:
            self.position.close()

class RsiStrategy(Strategy):
    """RSI超买超卖策略"""
    Name = "RSI策略"
    
    def init(self):
        self.rsi = self.I(calculate_rsi, self.data.Close, 14)
    
    def next(self):
        if self.rsi[-1] < 30 and not self.position:
            self.buy()
        elif self.rsi[-1] > 70 and self.position:
            self.position.close()

class MacdStrategy(Strategy):
    """MACD交叉策略"""
    Name = "MACD策略"
    
    def init(self):
        macd_line, signal_line = calculate_macd(pd.Series(self.data.Close), 12, 26, 9)
        self.macd = self.I(lambda: macd_line)
        self.signal = self.I(lambda: signal_line)
    
    def next(self):
        if crossover(self.macd, self.signal) and not self.position:
            self.buy()
        elif crossover(self.signal, self.macd) and self.position:
            self.position.close()

class BollingerStrategy(Strategy):
    """布林带策略"""
    Name = "布林带策略"
    
    def init(self):
        upper, middle, lower = calculate_bollinger_bands(pd.Series(self.data.Close), 20, 2)
        self.upper = self.I(lambda: upper)
        self.lower = self.I(lambda: lower)
    
    def next(self):
        if self.data.Close[-1] < self.lower[-1] and not self.position:
            self.buy()
        elif self.data.Close[-1] > self.upper[-1] and self.position:
            self.position.close()

class KdjStrategy(Strategy):
    """KDJ策略"""
    Name = "KDJ策略"
    
    def init(self):
        k, d = calculate_stochastic(
            pd.Series(self.data.High),
            pd.Series(self.data.Low), 
            pd.Series(self.data.Close),
            14, 3
        )
        self.k = self.I(lambda: k)
        self.d = self.I(lambda: d)
    
    def next(self):
        if crossover(self.k, self.d) and self.k[-1] < 20 and not self.position:
            self.buy()
        elif crossover(self.d, self.k) and self.k[-1] > 80 and self.position:
            self.position.close()

# ------------------------------------------------------------------
# 4. 数据获取和回测函数
# ------------------------------------------------------------------
def download_data(ticker, start_date, end_date):
    """下载股票数据"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None
        
        # 清理数据
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 确保有必要的列
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            return None
        
        df = df[required].dropna()
        
        if len(df) < 50:
            return None
            
        return df
    except:
        return None

def run_single_backtest(strategy_class, ticker, name):
    """运行单个回测"""
    try:
        # 下载数据
        df = download_data(
            ticker, 
            CONFIG["BACKTEST"]["start_date"],
            CONFIG["BACKTEST"]["end_date"]
        )
        
        if df is None:
            return None
        
        # 运行回测
        bt = Backtest(
            df,
            strategy_class,
            cash=CONFIG["BACKTEST"]["initial_cash"],
            commission=CONFIG["BACKTEST"]["commission"]
        )
        
        stats = bt.run()
        
        # 生成简单的图表（使用最小配置）
        safe_ticker = ticker.replace("^", "").replace(".", "_").replace("-", "_")
        filename = f"{strategy_class.Name}_{safe_ticker}.html"
        filepath = os.path.join("public", "reports", filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 生成图表（简化配置，避免错误）
        try:
            bt.plot(
                filename=filepath,
                open_browser=False,
                plot_volume=False,
                plot_drawdown=False,
                superimpose=False,
                plot_pl=False,
                plot_return=False
            )
        except Exception as e:
            print(f"    图表生成失败: {e}")
            # 创建简单的HTML占位符
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'''
                <!DOCTYPE html>
                <html>
                <head><title>{strategy_class.Name} - {name}</title></head>
                <body>
                    <h1>回测图表</h1>
                    <p>策略: {strategy_class.Name}</p>
                    <p>标的: {name} ({ticker})</p>
                    <p>交易次数: {stats.get('# Trades', 0)}</p>
                    <p>最终净值: ${stats.get('Equity Final [$]', 0):.2f}</p>
                    <p>总收益率: {stats.get('Return [%]', 0):.2f}%</p>
                    <p>注: 图表生成失败，请查看统计数据</p>
                </body>
                </html>
                ''')
        
        # 准备统计信息
        stats_info = {
            "标的名称": name,
            "标的代码": ticker,
            "策略名称": strategy_class.Name,
            "数据起点": str(df.index[0].date()),
            "数据终点": str(df.index[-1].date()),
            "数据条数": len(df),
            "交易次数": stats.get('# Trades', 0),
            "最终净值": stats.get('Equity Final [$]', 0),
            "总收益率": stats.get('Return [%]', 0),
            "年化收益率": stats.get('Return (Ann.) [%]', 0),
            "最大回撤": stats.get('Max. Drawdown [%]', 0),
            "夏普比率": stats.get('Sharpe Ratio', 0),
            "胜率": stats.get('Win Rate [%]', 0),
            "盈利因子": stats.get('Profit Factor', 0),
        }
        
        return {
            "file": f"reports/{filename}",
            "stats": stats_info
        }
        
    except Exception as e:
        print(f"    回测失败: {str(e)[:50]}")
        return None

# ------------------------------------------------------------------
# 5. 主程序
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("📊 多策略回测系统")
    print("=" * 60)
    
    # 创建目录
    os.makedirs("public", exist_ok=True)
    os.makedirs("public/reports", exist_ok=True)
    
    # 策略列表
    strategies = [
        SmaStrategy,
        RsiStrategy,
        MacdStrategy,
        BollingerStrategy,
        KdjStrategy
    ]
    
    # 存储结果
    results = {}
    all_reports = []
    
    for strategy_class in strategies:
        strategy_name = strategy_class.Name
        results[strategy_name] = {}
        
        print(f"\n📈 策略: {strategy_name}")
        print("-" * 40)
        
        for ticker, name in CONFIG["STOCKS"].items():
            print(f"  {name} ({ticker})...", end=" ")
            
            result = run_single_backtest(strategy_class, ticker, name)
            
            if result:
                results[strategy_name][ticker] = result
                
                # 添加到报告列表
                all_reports.append({
                    "策略": strategy_name,
                    "标的代码": ticker,
                    "标的名称": name,
                    "年化收益%": result["stats"]["年化收益率"],
                    "夏普比率": result["stats"]["夏普比率"],
                    "最大回撤%": result["stats"]["最大回撤"],
                    "总收益率%": result["stats"]["总收益率"],
                    "胜率%": result["stats"]["胜率"],
                    "交易次数": result["stats"]["交易次数"],
                    "盈利因子": result["stats"]["盈利因子"],
                    "报告文件": result["file"],
                })
                
                print(f"✅ {result['stats']['交易次数']}笔交易")
            else:
                print("❌")
    
    print(f"\n🎉 回测完成")
    
    # 生成CSV报告
    if all_reports:
        df = pd.DataFrame(all_reports)
        df.to_csv("public/strategy_comparison.csv", index=False, encoding='utf-8-sig')
        print(f"📊 CSV报告已生成")
    
    # 生成HTML页面
    generate_html(results, "public")
    
    print("\n" + "=" * 60)
    print("✅ 所有任务完成!")
    print("📁 输出目录: public/")
    print("🌐 请打开 public/index.html 查看结果")
    print("=" * 60)
    
    return True

def generate_html(results, output_dir):
    """生成HTML主页面"""
    
    # 构建下拉选项
    strategy_options = ""
    for strategy_name in results.keys():
        strategy_options += f'<option value="{strategy_name}">{strategy_name}</option>'
    
    stock_options = ""
    for ticker, name in CONFIG["STOCKS"].items():
        stock_options += f'<option value="{ticker}">{name} ({ticker})</option>'
    
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
        .btn-group {{
            display: flex;
            gap: 15px;
            justify-content: center;
        }}
        .btn {{
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
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
            border: 1px solid #ddd;
        }}
        .chart-frame {{
            width: 100%;
            height: 700px;
            border: none;
            display: block;
        }}
        .stats-sidebar {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            max-height: 700px;
            border: 1px solid #ddd;
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
            <p>支持5大技术指标策略：SMA, RSI, MACD, 布林带, KDJ</p>
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
                    加载回测报告
                </button>
                <button class="btn btn-secondary" onclick="downloadCSV()">
                    下载完整报告
                </button>
            </div>
        </div>
        
        <div class="content">
            <div class="chart-container">
                <!-- 使用object标签替代iframe，兼容性更好 -->
                <object id="chart-frame" class="chart-frame" 
                        type="text/html"
                        data="about:blank">
                    您的浏览器不支持内嵌HTML显示。
                </object>
            </div>
            
            <div class="stats-sidebar">
                <h2 style="color: #1a2980; margin-bottom: 20px;">📊 性能指标</h2>
                <table class="stats-table" id="stats-table">
                    <tbody>
                        <tr><td>标的名称</td><td id="stat-name">--</td></tr>
                        <tr><td>数据期间</td><td id="stat-period">--</td></tr>
                        <tr><td>数据条数</td><td id="stat-count">--</td></tr>
                        <tr><td>交易次数</td><td id="stat-trades">--</td></tr>
                        <tr><td>最终净值</td><td id="stat-final">--</td></tr>
                        <tr><td>总收益率</td><td id="stat-return">--</td></tr>
                        <tr><td>年化收益率</td><td id="stat-annual">--</td></tr>
                        <tr><td>最大回撤</td><td id="stat-drawdown">--</td></tr>
                        <tr><td>夏普比率</td><td id="stat-sharpe">--</td></tr>
                        <tr><td>胜率</td><td id="stat-winrate">--</td></tr>
                    </tbody>
                </table>
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
                // 使用object标签加载图表
                const chartFrame = document.getElementById('chart-frame');
                chartFrame.data = report.file;
                
                // 更新统计数据
                updateStats(report.stats);
                
                // 显示成功通知
                showNotification('✅ 报告加载成功', 'success');
            }} else {{
                document.getElementById('chart-frame').data = 'about:blank';
                clearStats();
                showNotification('❌ 未找到回测报告', 'error');
            }}
        }}
        
        function updateStats(stats) {{
            const formatNumber = (num, decimals = 2) => {{
                if (num == null || num === undefined) return '--';
                if (typeof num === 'number') return num.toFixed(decimals);
                return num;
            }};
            
            document.getElementById('stat-name').textContent = stats.标的名称 || '--';
            document.getElementById('stat-period').textContent = 
                `${{stats.数据起点 || '--'}} 至 ${{stats.数据终点 || '--'}}`;
            document.getElementById('stat-count').textContent = stats.数据条数 || '--';
            document.getElementById('stat-trades').textContent = stats.交易次数 || '--';
            document.getElementById('stat-final').textContent = stats.最终净值 ? '$' + formatNumber(stats.最终净值) : '--';
            document.getElementById('stat-return').textContent = stats.总收益率 ? formatNumber(stats.总收益率) + '%' : '--';
            document.getElementById('stat-annual').textContent = stats.年化收益率 ? formatNumber(stats.年化收益率) + '%' : '--';
            document.getElementById('stat-drawdown').textContent = stats.最大回撤 ? formatNumber(stats.最大回撤) + '%' : '--';
            document.getElementById('stat-sharpe').textContent = stats.夏普比率 ? formatNumber(stats.夏普比率) : '--';
            document.getElementById('stat-winrate').textContent = stats.胜率 ? formatNumber(stats.胜率) + '%' : '--';
            
            // 高亮显示关键指标
            highlightStats(stats);
        }}
        
        function highlightStats(stats) {{
            const highlight = (id, condition) => {{
                const el = document.getElementById(id);
                if (el && condition) {{
                    el.style.color = '#28a745';
                    el.style.fontWeight = 'bold';
                }} else if (el) {{
                    el.style.color = '';
                    el.style.fontWeight = '';
                }}
            }};
            
            highlight('stat-sharpe', stats.夏普比率 > 1);
            highlight('stat-winrate', stats.胜率 > 50);
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
                setTimeout(() => {{
                    if (notification.parentNode) {{
                        notification.parentNode.removeChild(notification);
                    }}
                }}, 300);
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
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 主页面已生成: {output_dir}/index.html")

# ------------------------------------------------------------------
# 6. 程序入口
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
