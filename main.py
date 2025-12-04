#!/usr/bin/env python3
# main.py - 简化的多策略回测系统

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

# 配置参数
STOCKS_CONFIG = {
    "港股": {
        "^HSI": "恒生指数",
        "0700.HK": "腾讯控股",
        "9988.HK": "阿里巴巴-SW",
        "3690.HK": "美团-W",
    },
    "美股": {
        "SPY": "标普500 ETF",
        "QQQ": "纳指100 ETF",
        "AAPL": "苹果",
        "MSFT": "微软",
        "GOOGL": "谷歌",
    },
}

BACKTEST_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",
    "initial_cash": 100000,
    "commission": 0.002,
}

# 策略定义（简化版）
class SimpleStrategy(Strategy):
    Name = "简单策略"
    
    def init(self):
        # 使用简单的移动平均线
        self.sma20 = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.Close)
        self.sma50 = self.I(lambda x: pd.Series(x).rolling(50).mean(), self.data.Close)
    
    def next(self):
        if crossover(self.sma20, self.sma50):
            if not self.position:
                self.buy()
        elif crossover(self.sma50, self.sma20):
            if self.position:
                self.position.close()

class RsiStrategy(Strategy):
    Name = "RSI策略"
    
    def init(self):
        # 使用pandas计算RSI
        delta = self.data.Close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        self.rsi = self.I(lambda: rsi)
    
    def next(self):
        if self.rsi[-1] < 30 and not self.position:
            self.buy()
        elif self.rsi[-1] > 70 and self.position:
            self.position.close()

def get_data(ticker, start="2022-01-01", end="2023-12-31"):
    """获取股票数据"""
    print(f"下载 {ticker} 数据...")
    try:
        # 使用yfinance下载
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            print(f"  ⚠️  无数据")
            return None
        
        # 确保列名正确
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 重命名列（如果需要）
        column_mapping = {
            'Adj Close': 'Close',
            'adjclose': 'Close',
            'Adj Close': 'Close'
        }
        df = df.rename(columns=column_mapping)
        
        # 确保有必要的列
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            print(f"  ⚠️  缺少必要列")
            return None
        
        print(f"  ✅  {len(df)} 条数据")
        return df
    
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return None

def main():
    print("=" * 60)
    print("📊 多策略回测系统")
    print("=" * 60)
    
    # 输出目录
    OUT_DIR = "public"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)
    
    # 策略列表
    STRATEGIES = [SimpleStrategy, RsiStrategy]
    
    # 存储结果
    results = {}
    
    for strategy in STRATEGIES:
        strategy_name = strategy.Name
        results[strategy_name] = {}
        print(f"\n📈 运行策略: {strategy_name}")
        
        for ticker, name in STOCKS_CONFIG["港股"].items():
            print(f"\n  处理 {name} ({ticker})...")
            
            # 获取数据
            data = get_data(ticker)
            if data is None or len(data) < 100:
                print(f"  ⏭️  跳过，数据不足")
                continue
            
            try:
                # 运行回测
                bt = Backtest(
                    data,
                    strategy,
                    cash=BACKTEST_CONFIG["initial_cash"],
                    commission=BACKTEST_CONFIG["commission"]
                )
                
                # 运行策略
                stats = bt.run()
                print(f"  ✅  回测完成")
                print(f"     交易次数: {stats['# Trades']}")
                print(f"     最终净值: ${stats['Equity Final [$]']:.2f}")
                print(f"     总收益率: {stats['Return [%]']:.2f}%")
                
                # 生成HTML报告
                safe_ticker = ticker.replace("^", "").replace(".", "_").replace("-", "_")
                filename = f"{strategy_name}_{safe_ticker}.html"
                filepath = os.path.join(OUT_DIR, "reports", filename)
                
                # 生成图表（简化版）
                bt.plot(
                    filename=filepath,
                    open_browser=False,
                    plot_volume=False,
                    plot_drawdown=True
                )
                
                # 保存统计数据
                stats_dict = {
                    k: v for k, v in stats.items() 
                    if isinstance(v, (int, float, str, bool)) and not k.startswith('_')
                }
                
                # 添加额外信息
                stats_dict.update({
                    "标的名称": name,
                    "数据起点": str(data.index[0].date()),
                    "数据终点": str(data.index[-1].date()),
                    "数据条数": len(data),
                })
                
                # 存储结果
                results[strategy_name][ticker] = {
                    "file": f"reports/{filename}",
                    "stats": stats_dict
                }
                
                print(f"  📄  报告生成: {filename}")
                
            except Exception as e:
                print(f"  ❌  回测失败: {e}")
                continue
    
    # 生成主页面
    generate_html(results, OUT_DIR)
    
    print("\n" + "=" * 60)
    print("🎉 所有回测完成!")
    print(f"📁 输出目录: {OUT_DIR}")
    print("=" * 60)
    return True

def generate_html(results, out_dir):
    """生成HTML主页面"""
    
    # 构建下拉选项
    strategy_options = ""
    for strategy_name in results.keys():
        strategy_options += f'<option value="{strategy_name}">{strategy_name}</option>\n'
    
    ticker_options = ""
    for market, tickers in STOCKS_CONFIG.items():
        ticker_options += f'<optgroup label="{market}">\n'
        for ticker, name in tickers.items():
            ticker_options += f'<option value="{ticker}">{name} ({ticker})</option>\n'
        ticker_options += '</optgroup>\n'
    
    # 转换结果为JSON
    results_json = json.dumps(results, ensure_ascii=False, indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多策略回测系统</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Microsoft JhengHei', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        }}
        .controls {{
            padding: 25px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .control-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .control-item {{
            flex: 1;
            min-width: 250px;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }}
        select {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ced4da;
            border-radius: 8px;
            font-size: 16px;
            background: white;
        }}
        select:focus {{
            border-color: #667eea;
            outline: none;
        }}
        .btn {{
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .content {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            padding: 25px;
            min-height: 600px;
        }}
        .report-container {{
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            background: white;
        }}
        .report-frame {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        .stats-panel {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            max-height: 600px;
        }}
        .stats-panel h3 {{
            margin-bottom: 20px;
            color: #333;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats-table th, .stats-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .stats-table th {{
            font-weight: 600;
            color: #495057;
            background: #e9ecef;
        }}
        .stats-table tr:hover {{
            background: #f1f3f5;
        }}
        .footer {{
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            background: #f8f9fa;
        }}
        @media (max-width: 768px) {{
            .content {{ grid-template-columns: 1fr; }}
            .report-frame {{ height: 400px; }}
            .stats-panel {{ max-height: 400px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 多策略回测系统</h1>
            <p>覆盖港股、美股，支持多种技术指标策略</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <div class="control-item">
                    <label>选择策略:</label>
                    <select id="strategy-select">
                        {strategy_options}
                    </select>
                </div>
                
                <div class="control-item">
                    <label>选择标的:</label>
                    <select id="ticker-select">
                        {ticker_options}
                    </select>
                </div>
            </div>
            
            <div style="text-align: center;">
                <button class="btn" onclick="loadReport()">
                    🔍 加载回测报告
                </button>
            </div>
        </div>
        
        <div class="content">
            <div class="report-container">
                <iframe id="report-frame" class="report-frame" 
                        title="回测报告"
                        src="about:blank">
                </iframe>
            </div>
            
            <div class="stats-panel">
                <h3>📈 性能指标</h3>
                <table class="stats-table" id="stats-table">
                    <tbody id="stats-body">
                        <tr><td>数据起点</td><td id="start-date">--</td></tr>
                        <tr><td>数据终点</td><td id="end-date">--</td></tr>
                        <tr><td>数据条数</td><td id="data-count">--</td></tr>
                        <tr><td>交易次数</td><td id="trade-count">--</td></tr>
                        <tr><td>最终净值</td><td id="equity-final">--</td></tr>
                        <tr><td>总收益率</td><td id="return-pct">--</td></tr>
                        <tr><td>年化收益率</td><td id="return-ann">--</td></tr>
                        <tr><td>夏普比率</td><td id="sharpe">--</td></tr>
                        <tr><td>最大回撤</td><td id="max-dd">--</td></tr>
                        <tr><td>胜率</td><td id="win-rate">--</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>📅 数据更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>⚠️ 风险提示: 回测结果基于历史数据，不代表未来表现</p>
        </div>
    </div>
    
    <script>
        // 回测数据
        const RESULTS = {results_json};
        
        // 页面加载完成后设置默认值
        window.onload = function() {{
            if (Object.keys(RESULTS).length > 0) {{
                // 设置默认策略
                const firstStrategy = Object.keys(RESULTS)[0];
                document.getElementById('strategy-select').value = firstStrategy;
                
                // 设置默认标的
                const firstTicker = Object.keys(RESULTS[firstStrategy])[0];
                if (firstTicker) {{
                    document.getElementById('ticker-select').value = firstTicker;
                    loadReport();
                }}
            }}
        }};
        
        function loadReport() {{
            const strategy = document.getElementById('strategy-select').value;
            const ticker = document.getElementById('ticker-select').value;
            
            // 获取报告信息
            const reportInfo = RESULTS[strategy]?.[ticker];
            
            if (reportInfo && reportInfo.file) {{
                // 加载报告到iframe
                const iframe = document.getElementById('report-frame');
                iframe.src = reportInfo.file;
                
                // 更新统计数据
                updateStats(reportInfo.stats);
                
                // 显示成功消息
                showMessage('✅ 报告加载成功', 'success');
            }} else {{
                // 清空iframe
                document.getElementById('report-frame').src = 'about:blank';
                
                // 清空统计数据
                clearStats();
                
                // 显示错误消息
                showMessage('❌ 未找到回测报告', 'error');
            }}
        }}
        
        function updateStats(stats) {{
            // 更新表格数据
            document.getElementById('start-date').textContent = stats['数据起点'] || '--';
            document.getElementById('end-date').textContent = stats['数据终点'] || '--';
            document.getElementById('data-count').textContent = stats['数据条数'] || '--';
            document.getElementById('trade-count').textContent = stats['# Trades'] || '--';
            document.getElementById('equity-final').textContent = stats['Equity Final [$]'] ? '$' + stats['Equity Final [$]'].toFixed(2) : '--';
            document.getElementById('return-pct').textContent = stats['Return [%]'] ? stats['Return [%]'].toFixed(2) + '%' : '--';
            document.getElementById('return-ann').textContent = stats['Return (Ann.) [%]'] ? stats['Return (Ann.) [%]'].toFixed(2) + '%' : '--';
            document.getElementById('sharpe').textContent = stats['Sharpe Ratio'] ? stats['Sharpe Ratio'].toFixed(2) : '--';
            document.getElementById('max-dd').textContent = stats['Max. Drawdown [%]'] ? stats['Max. Drawdown [%]'].toFixed(2) + '%' : '--';
            document.getElementById('win-rate').textContent = stats['Win Rate [%]'] ? stats['Win Rate [%]'].toFixed(2) + '%' : '--';
            
            // 高亮好的指标
            highlightGoodStats(stats);
        }}
        
        function highlightGoodStats(stats) {{
            // 高亮夏普比率 > 1
            const sharpeEl = document.getElementById('sharpe');
            if (stats['Sharpe Ratio'] > 1) {{
                sharpeEl.style.color = '#28a745';
                sharpeEl.style.fontWeight = 'bold';
            }} else {{
                sharpeEl.style.color = '';
                sharpeEl.style.fontWeight = '';
            }}
            
            // 高亮胜率 > 50%
            const winRateEl = document.getElementById('win-rate');
            if (stats['Win Rate [%]'] > 50) {{
                winRateEl.style.color = '#28a745';
                winRateEl.style.fontWeight = 'bold';
            }} else {{
                winRateEl.style.color = '';
                winRateEl.style.fontWeight = '';
            }}
            
            // 高亮最大回撤 < -20%
            const maxDdEl = document.getElementById('max-dd');
            if (stats['Max. Drawdown [%]'] < -20) {{
                maxDdEl.style.color = '#dc3545';
                maxDdEl.style.fontWeight = 'bold';
            }} else {{
                maxDdEl.style.color = '';
                maxDdEl.style.fontWeight = '';
            }}
        }}
        
        function clearStats() {{
            const statCells = document.querySelectorAll('#stats-body td:nth-child(2)');
            statCells.forEach(cell => {{
                cell.textContent = '--';
                cell.style.color = '';
                cell.style.fontWeight = '';
            }});
        }}
        
        function showMessage(message, type) {{
            // 创建消息元素
            const msgEl = document.createElement('div');
            msgEl.textContent = message;
            msgEl.style.cssText = `
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
            
            // 设置颜色
            if (type === 'success') {{
                msgEl.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
            }} else if (type === 'error') {{
                msgEl.style.background = 'linear-gradient(135deg, #dc3545 0%, #fd7e14 100%)';
            }}
            
            document.body.appendChild(msgEl);
            
            // 3秒后移除
            setTimeout(() => {{
                msgEl.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => msgEl.remove(), 300);
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
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 主页面已生成: {out_dir}/index.html")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
