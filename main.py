#!/usr/bin/env python3
# main.py – 多策略回测系统（TA-Lib 版）
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1. 配置
# ------------------------------------------------------------------
CONFIG = {
    "STOCKS": {
        "^HSI": "恒生指数", "0700.HK": "腾讯控股", "9988.HK": "阿里巴巴",
        "AAPL": "苹果", "MSFT": "微软", "GOOGL": "谷歌",
        "TSLA": "特斯拉", "NVDA": "英伟达", "SPY": "标普500 ETF", "QQQ": "纳指100 ETF",
    },
    "BACKTEST": {
        "start_date": (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        "end_date": datetime.today().strftime("%Y-%m-%d"),
        "initial_cash": 100_000,
        "commission": 0.002,
    },
    "STRATEGY_PARAMS": {
        "SMA": {"fast": 10, "slow": 30},
        "RSI": {"period": 14, "oversold": 30, "overbought": 70},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BB": {"period": 20, "std_dev": 2},
        "KDJ": {"kp": 14, "dp": 3},
    },
}

# ------------------------------------------------------------------
# 2. 指标（TA-Lib）
# ------------------------------------------------------------------
def sma(close, n): return ta.SMA(close, n)
def ema(close, n): return ta.EMA(close, n)
def rsi(close, n=14): return ta.RSI(close, n)
def macd_ext(close, f=12, s=26, sig=9):
    macd, signal, hist = ta.MACD(close, fastperiod=f, slowperiod=s, signalperiod=sig)
    return macd, signal, hist
def bbands(close, n=20, d=2):
    upper, mid, lower = ta.BBANDS(close, n, d, d)
    return upper, mid, lower
def stochastic(high, low, close, kp=14, dp=3):
    k, d = ta.STOCH(high, low, close, fastk_period=kp, slowk_period=dp, slowd_period=dp)
    j = 3 * k - 2 * d
    return k, d, j

# ------------------------------------------------------------------
# 3. 策略
# ------------------------------------------------------------------
class SmaStrategy(Strategy):
    Name = "SMA策略"
    def init(self):
        p = CONFIG["STRATEGY_PARAMS"]["SMA"]
        fast = self.I(sma, self.data.Close, p["fast"])
        slow = self.I(sma, self.data.Close, p["slow"])
        self.buy_sig = crossover(fast, slow)
        self.sell_sig = crossover(slow, fast)
    def next(self):
        if self.buy_sig: self.buy()
        elif self.sell_sig: self.position.close()

class RsiStrategy(Strategy):
    Name = "RSI策略"
    def init(self):
        p = CONFIG["STRATEGY_PARAMS"]["RSI"]
        self.r = self.I(rsi, self.data.Close, p["period"])
        self.o, self.b = p["overbought"], p["oversold"]
    def next(self):
        if self.r[-1] < self.b and not self.position: self.buy()
        elif self.r[-1] > self.o and self.position: self.position.close()

class MacdStrategy(Strategy):
    Name = "MACD策略"
    def init(self):
        p = CONFIG["STRATEGY_PARAMS"]["MACD"]
        macd_line, signal_line, _ = self.I(
            macd_ext, self.data.Close, p["fast"], p["slow"], p["signal"])
        self.macd, self.signal = macd_line, signal_line
    def next(self):
        if crossover(self.macd, self.signal): self.buy()
        elif crossover(self.signal, self.macd): self.position.close()

class BollingerBandsStrategy(Strategy):
    Name = "布林带策略"
    def init(self):
        p = CONFIG["STRATEGY_PARAMS"]["BB"]
        upper, _, lower = self.I(bbands, self.data.Close, p["period"], p["std_dev"])
        self.u, self.l = upper, lower
    def next(self):
        price = self.data.Close[-1]
        if price < self.l[-1] and not self.position: self.buy()
        elif price > self.u[-1] and self.position: self.position.close()

class KdjStrategy(Strategy):
    Name = "KDJ策略"
    def init(self):
        p = CONFIG["STRATEGY_PARAMS"]["KDJ"]
        k, d, _ = self.I(stochastic, self.data.High, self.data.Low, self.data.Close,
                         p["kp"], p["dp"])
        self.k, self.d = k, d
    def next(self):
        if crossover(self.k, self.d) and self.k[-1] < 20 and not self.position: self.buy()
        elif crossover(self.d, self.k) and self.k[-1] > 80 and self.position: self.position.close()

# ------------------------------------------------------------------
# 4. 工具
# ------------------------------------------------------------------
def fetch(tic, start, end):
    try:
        df = yf.download(tic, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return None if len(df) < 30 else df
    except Exception as e:
        print(f" ❌ 下载失败 {tic}: {e}")
        return None

def safe(tic): return tic.replace("^", "").replace(".", "_").replace("-", "_")

# ------------------------------------------------------------------
# 5. 回测
# ------------------------------------------------------------------
def run_single(strategy_cls, tic, name):
    df = fetch(tic, CONFIG["BACKTEST"]["start_date"], CONFIG["BACKTEST"]["end_date"])
    if df is None: return None
    bt = Backtest(df, strategy_cls,
                  cash=CONFIG["BACKTEST"]["initial_cash"],
                  commission=CONFIG["BACKTEST"]["commission"])
    stats = bt.run()
    os.makedirs("public/reports", exist_ok=True)
    report_path = f"reports/{strategy_cls.Name}_{safe(tic)}.html"
    bt.plot(filename=f"public/{report_path}", open_browser=False, plot_volume=False)
    stats_dict = {k: v for k, v in stats.items() if isinstance(v, (int, float, str)) and not k.startswith('_')}
    stats_dict.update({
        "标的名称": name,
        "标的代码": tic,
        "策略名称": strategy_cls.Name,
        "数据起点": str(df.index[0].date()),
        "数据终点": str(df.index[-1].date()),
        "数据条数": len(df),
        "初始资金": CONFIG["BACKTEST"]["initial_cash"],
        "手续费率": CONFIG["BACKTEST"]["commission"],
    })
    return {"file": report_path, "stats": stats_dict}

# ------------------------------------------------------------------
# 6. 主程序
# ------------------------------------------------------------------
def main():
    print("📊 多策略回测系统（TA-Lib 版）")
    os.makedirs("public/reports", exist_ok=True)
    strategies = [SmaStrategy, RsiStrategy, MacdStrategy, BollingerBandsStrategy, KdjStrategy]
    results, records = {}, []

    for st in strategies:
        print(f"\n📈 策略：{st.Name}")
        results[st.Name] = {}
        for tic, name in CONFIG["STOCKS"].items():
            print(f"  {name} ({tic}) ...", end="")
            ret = run_single(st, tic, name)
            if ret:
                results[st.Name][tic] = ret
                records.append({
                    "策略": st.Name,
                    "标的代码": tic,
                    "标的名称": name,
                    "年化收益%": ret["stats"].get("Return (Ann.) [%]", 0),
                    "夏普比率": ret["stats"].get("Sharpe Ratio", 0),
                    "最大回撤%": ret["stats"].get("Max. Drawdown [%]", 0),
                    "总收益率%": ret["stats"].get("Return [%]", 0),
                    "胜率%": ret["stats"].get("Win Rate [%]", 0),
                    "交易次数": ret["stats"].get("# Trades", 0),
                    "盈利因子": ret["stats"].get("Profit Factor", 0),
                    "报告文件": ret["file"],
                })
                print(" ✅")
            else:
                print(" ❌")

    if records:
        pd.DataFrame(records).sort_values("夏普比率", ascending=False).to_csv("public/strategy_comparison.csv", index=False, encoding="utf-8-sig")
        print("\n📊 已生成 strategy_comparison.csv")
    generate_html(results, "public")
    print("\n✅ 全部完成！请打开 public/index.html 查看结果")

# ------------------------------------------------------------------
# 7. 生成主页
# ------------------------------------------------------------------
def generate_html(results, out_dir):
    strategy_opts = "\n".join([f'<option value="{s}">{s}</option>' for s in results])
    stock_opts = "\n".join([f'<option value="{t}">{n} ({t})</option>' for t, n in CONFIG["STOCKS"].items()])
    best = ""
    try:
        df = pd.read_csv("public/strategy_comparison.csv")
        if not df.empty:
            b = df.iloc[0]
            best = f'<div class="recommendations"><h3>🏆 最佳组合</h3><p><strong>{b["策略"]} + {b["标的名称"]}</strong></p><p>夏普 {b["夏普比率"]:.2f} | 年化 {b["年化收益%"]:.1f}% | 回撤 {b["最大回撤%"]:.1f}%</p></div>'
    except: pass
    results_json = json.dumps(results, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>多策略回测系统</title>
<style>
body{{font-family:system-ui, sans-serif; margin:0; background:#f2f6ff}}
.header{{background:#0d47a1; color:white; padding:30px; text-align:center}}
.controls{{display:flex; gap:15px; justify-content:center; padding:20px; background:#e3f2fd}}
select{{padding:8px 12px; font-size:16px}}
.btn{{padding:10px 20px; background:#0d47a1; color:white; border:none; border-radius:4px; cursor:pointer}}
.btn:hover{{opacity:.9}}
.content{{display:grid; grid-template-columns:1fr 350px; gap:20px; padding:20px}}
.chart-frame{{width:100%; height:700px; border:1px solid #ddd; background:white}}
.stats-sidebar{{background:white; padding:20px; border:1px solid #ddd; border-radius:6px}}
.stats-table{{width:100%; border-collapse:collapse}}
.stats-table th,.stats-table td{{padding:8px; border-bottom:1px solid #eee; text-align:left}}
@media(max-width:900px){{.content{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="header">
  <h1>📊 多策略回测系统</h1>
  <p>支持 SMA、RSI、MACD、布林带、KDJ 五种技术指标</p>
</div>
<div class="controls">
  <select id="strategy">{strategy_opts}</select>
  <select id="stock">{stock_opts}</select>
  <button class="btn" onclick="loadReport()">加载回测报告</button>
  <button class="btn" onclick="downloadCSV()">下载完整 CSV</button>
</div>
<div class="content">
  <iframe id="chart" class="chart-frame" title="回测图表" src="about:blank"></iframe>
  <div class="stats-sidebar">
    <h3>📊 性能指标</h3>
    <table class="stats-table" id="stats"></table>
    {best}
  </div>
</div>
<script>
const DATA = {results_json};
function loadReport() {{
  const s = document.getElementById('strategy').value;
  const t = document.getElementById('stock').value;
  const item = DATA[s]?.[t];
  if (!item) return alert('未找到报告');
  document.getElementById('chart').src = item.file;
  const st = item.stats;
  const rows = [
    ['标的名称', st['标的名称']],
    ['数据期间', `${{st['数据起点']}} 至 ${{st['数据终点']}}`],
    ['总收益率', (st['Return [%]'] || 0).toFixed(2) + '%'],
    ['年化收益率', (st['Return (Ann.) [%]'] || 0).toFixed(2) + '%'],
    ['夏普比率', (st['Sharpe Ratio'] || 0).toFixed(2)],
    ['最大回撤', (st['Max. Drawdown [%]'] || 0).toFixed(2) + '%'],
    ['交易次数', st['# Trades']],
    ['胜率', (st['Win Rate [%]'] || 0).toFixed(1) + '%'],
  ];
  document.getElementById('stats').innerHTML = rows.map(([k,v])=>`<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
}}
function downloadCSV() {{ window.open('strategy_comparison.csv', '_blank'); }}
window.onload = () => document.querySelector('button').click();
</script>
</body>
</html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
