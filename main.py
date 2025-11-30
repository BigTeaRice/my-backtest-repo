#!/usr/bin/env python3
# main.py
import os
import json
import yfinance as yf
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ------------------------------------------------------------------
# 1. 策略定义（保持不变）
# ------------------------------------------------------------------
class RsiOscillator(Strategy):
    Name = "RSI_Oscillator"
    upper, lower, window = 70, 30, 14
    def init(self): self.rsi = self.I(talib.RSI, self.data.Close, self.window)
    def next(self):
        if crossover(self.rsi, self.upper): self.position.close()
        elif crossover(self.lower, self.rsi): self.buy()

class SmaCrossover(Strategy):
    Name = "SMA_Crossover"
    fast, slow = 20, 50
    def init(self):
        self.sma_f = self.I(talib.SMA, self.data.Close, self.fast)
        self.sma_s = self.I(talib.SMA, self.data.Close, self.slow)
    def next(self):
        if crossover(self.sma_f, self.sma_s): self.buy()
        elif crossover(self.sma_s, self.sma_f): self.position.close()

class MacdCrossover(Strategy):
    Name = "MACD_Crossover"
    fast, slow, signal = 12, 26, 9
    def init(self):
        self.macd, self.sig, _ = self.I(talib.MACD, self.data.Close,
                                        fastperiod=self.fast,
                                        slowperiod=self.slow,
                                        signalperiod=self.signal)
    def next(self):
        if crossover(self.macd, self.sig): self.buy()
        elif crossover(self.sig, self.macd): self.position.close()

# ------------------------------------------------------------------
# 2. 数据抓取
# ------------------------------------------------------------------
def get_data(ticker: str, start: str = "2020-01-01") -> pd.DataFrame:
    print(f"Fetching data for {ticker} from yfinance...")
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df if all(col in df.columns for col in ["Open","High","Low","Close","Volume"]) else pd.DataFrame()
    except Exception as e:
        print("YFinance error:", e)
        return pd.DataFrame()

# ------------------------------------------------------------------
# 3. 主程序
# ------------------------------------------------------------------
if __name__ == "__main__":
    TICKERS = {"^HSI": "恆生指數 (HSI)", "AAPL": "Apple (AAPL)", "TSLA": "Tesla (TSLA)"}
    STRATEGIES = [RsiOscillator, SmaCrossover, MacdCrossover]
    OUT_DIR = "public"
    os.makedirs(OUT_DIR, exist_ok=True)

    reports_map = {}   # {strategy: {ticker: filename}}

    for Stg in STRATEGIES:
        stg_name = Stg.Name
        reports_map[stg_name] = {}
        for tic, desc in TICKERS.items():
            data = get_data(tic)
            need = max(getattr(Stg, 'window', 0),
                       getattr(Stg, 'slow', 0),
                       getattr(Stg, 'signal', 0))
            if len(data) < need + 50:
                print(f"Skip {tic}/{stg_name}: data too short")
                continue
            try:
                bt = Backtest(data, Stg, cash=100_000, commission=.002)
                stats = bt.run()
                safe = tic.replace("^", "").replace(".", "_")
                fname = f"{stg_name}_{safe}.html"
                bt.plot(filename=os.path.join(OUT_DIR, fname), open_browser=False)
                reports_map[stg_name][tic] = fname
                # 把关键指标写进map，前端可直接用
                reports_map[stg_name][tic+"_stats"] = {
                    "Start": str(stats.get('Start', '')),
                    "End": str(stats.get('End', '')),
                    "Duration": str(stats.get('Duration', '')),
                    "Exposure Time [%]": float(stats.get('Exposure Time [%]', 0)),
                    "Equity Final [$]": float(stats.get('Equity Final [$]', 0)),
                    "Equity Peak [$]": float(stats.get('Equity Peak [$]', 0)),
                    "Return [%]": float(stats.get('Return [%]', 0)),
                    "Buy & Hold Return [%]": float(stats.get('Buy & Hold Return [%]', 0)),
                    "Return (Ann.) [%]": float(stats.get('Return (Ann.) [%]', 0)),
                    "Volatility (Ann.) [%]": float(stats.get('Volatility (Ann.) [%]', 0)),
                    "Sharpe Ratio": float(stats.get('Sharpe Ratio', 0)),
                    "Sortino Ratio": float(stats.get('Sortino Ratio', 0)),
                    "Calmar Ratio": float(stats.get('Calmar Ratio', 0)),
                    "Max. Drawdown [%]": float(stats.get('Max. Drawdown [%]', 0)),
                    "Avg. Drawdown [%]": float(stats.get('Avg. Drawdown [%]', 0)),
                    "Max. Drawdown Duration": str(stats.get('Max. Drawdown Duration', '')),
                    "Avg. Drawdown Duration": str(stats.get('Avg. Drawdown Duration', '')),
                    "# Trades": int(stats.get('# Trades', 0)),
                    "Win Rate [%]": float(stats.get('Win Rate [%]', 0)),
                    "Best Trade [%]": float(stats.get('Best Trade [%]', 0)),
                    "Worst Trade [%]": float(stats.get('Worst Trade [%]', 0)),
                    "Avg. Trade [%]": float(stats.get('Avg. Trade [%]', 0)),
                    "Max. Trade Duration": str(stats.get('Max. Trade Duration', '')),
                    "Avg. Trade Duration": str(stats.get('Avg. Trade Duration', '')),
                    "Profit Factor": float(stats.get('Profit Factor', 0)),
                    "Expectancy [%]": float(stats.get('Expectancy [%]', 0)),
                    "SQN": float(stats.get('SQN', 0)),
                }
                print(f"✅ {tic} + {stg_name} -> {fname}")
            except Exception as e:
                print(f"❌ backtest {tic}/{stg_name}: {e}")

    # ------------------------------------------------------------------
    # 4. 生成新的 index.html（含输入框+统计表）
    # ------------------------------------------------------------------
    def build_select_options():
        strategy_opts = "\n".join(f'<option value="{s.Name}">{s.Name.replace("_", " ")}</option>' for s in STRATEGIES)
        ticker_opts   = "\n".join(f'<option value="{k}">{v} ({k})</option>' for k, v in TICKERS.items())
        return strategy_opts, ticker_opts

    strategy_options, ticker_options = build_select_options()
    reports_json = json.dumps(reports_map)
    default_report = reports_map.get(STRATEGIES[0].Name, {}).get(next(iter(TICKERS.keys())), "")

    # 统计表格 HTML 片段
    stats_table = """
    <table class="stats">
      <thead><tr><th class="left">指標</th><th>數值</th></tr></thead>
      <tbody id="stats-body">
        <tr><td class="left">Start</td><td id="st_Start">--</td></tr>
        <tr><td class="left">End</td><td id="st_End">--</td></tr>
        <tr><td class="left">Duration</td><td id="st_Duration">--</td></tr>
        <tr><td class="left">Exposure Time [%]</td><td id="st_Exposure">--</td></tr>
        <tr><td class="left">Equity Final [$]</td><td id="st_EquityFinal">--</td></tr>
        <tr><td class="left">Equity Peak [$]</td><td id="st_EquityPeak">--</td></tr>
        <tr><td class="left">Return [%]</td><td id="st_Return">--</td></tr>
        <tr><td class="left">Buy & Hold Return [%]</td><td id="st_BuyHold">--</td></tr>
        <tr><td class="left">Return (Ann.) [%]</td><td id="st_ReturnAnn">--</td></tr>
        <tr><td class="left">Volatility (Ann.) [%]</td><td id="st_VolAnn">--</td></tr>
        <tr><td class="left">Sharpe Ratio</td><td id="st_Sharpe">--</td></tr>
        <tr><td class="left">Sortino Ratio</td><td id="st_Sortino">--</td></tr>
        <tr><td class="left">Calmar Ratio</td><td id="st_Calmar">--</td></tr>
        <tr><td class="left">Max. Drawdown [%]</td><td id="st_MaxDD">--</td></tr>
        <tr><td class="left">Avg. Drawdown [%]</td><td id="st_AvgDD">--</td></tr>
        <tr><td class="left">Max. Drawdown Duration</td><td id="st_MaxDDD">--</td></tr>
        <tr><td class="left">Avg. Drawdown Duration</td><td id="st_AvgDDD">--</td></tr>
        <tr><td class="left"># Trades</td><td id="st_Trades">--</td></tr>
        <tr><td class="left">Win Rate [%]</td><td id="st_WinRate">--</td></tr>
        <tr><td class="left">Best Trade [%]</td><td id="st_BestTrade">--</td></tr>
        <tr><td class="left">Worst Trade [%]</td><td id="st_WorstTrade">--</td></tr>
        <tr><td class="left">Avg. Trade [%]</td><td id="st_AvgTrade">--</td></tr>
        <tr><td class="left">Max. Trade Duration</td><td id="st_MaxTradeDur">--</td></tr>
        <tr><td class="left">Avg. Trade Duration</td><td id="st_AvgTradeDur">--</td></tr>
        <tr><td class="left">Profit Factor</td><td id="st_ProfitFactor">--</td></tr>
        <tr><td class="left">Expectancy [%]</td><td id="st_Expectancy">--</td></tr>
        <tr><td class="left">SQN</td><td id="st_SQN">--</td></tr>
      </tbody>
    </table>
    """

    index_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>多策略回測報告中心</title>
  <style>
    body{{font-family:'Inter',Arial,sans-serif;margin:0;padding:20px;background:#f4f7f9;}}
    .container{{max-width:1200px;margin:0 auto;background:white;padding:25px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,.08);}}
    .header-control{{margin-bottom:25px;padding-bottom:15px;border-bottom:2px solid #e0e0e0;}}
    h2{{margin-top:0;color:#333;}}
    label{{font-weight:bold;margin-right:10px;color:#555;}}
    select,input{{padding:10px 14px;font-size:15px;border:1px solid #ccc;border-radius:8px;margin-right:12px;}}
    button{{padding:10px 18px;font-size:15px;border:none;border-radius:8px;background:#007bff;color:#fff;cursor:pointer;}}
    button:hover{{background:#0056b3;}}
    #report-container{{width:100%;height:calc(100vh - 200px);border:none;border-radius:8px;margin-top:20px;}}
    .stats{{width:100%;border-collapse:collapse;font-size:14px;margin-top:30px;}}
    .stats th,.stats td{{border:1px solid #ddd;padding:6px 8px;text-align:right;}}
    .stats th{{background:#f8f9fa;text-align:center;}}
    .stats .left{{text-align:left;}}
  </style>
</head>
<body>
  <div class="container">
    <div class="header-control">
      <h2>多策略回測報告中心</h2>

      <!-- 股票代号查询 -->
      <div style="margin-bottom:15px;">
        <label>股票代號查詢：</label>
        <input id="symbol-input" type="text" placeholder="例：AAPL, ^HSI, TSLA">
        <button onclick="fetchSymbol()">查詢</button>
      </div>

      <!-- 原有下拉 -->
      <label for="strategy-select">選擇策略：</label>
      <select id="strategy-select" onchange="loadReport()">{strategy_options}</select>

      <label for="ticker-select">選擇股票代號：</label>
      <select id="ticker-select" onchange="loadReport()">{ticker_options}</select>

      <p class="note">報告包含夏普指數 (Sharpe Ratio)、最大回撤 (Max. Drawdown) 等關鍵指標。</p>
      <p class="note">資料來源：yfinance / 報告由 GitHub Actions 自動生成</p>
    </div>

    <!-- 回测报告 iframe -->
    <iframe id="report-container" src="{default_report}" frameborder="0"></iframe>

    <!-- 统计报告表格 -->
    {stats_table}
  </div>

  <script>
    const REPORTS_MAP = {reports_json};

    window.onload = () => loadReport();

    function loadReport() {{
      const st = document.getElementById('strategy-select').value;
      const tic = document.getElementById('ticker-select').value;
      const filename = REPORTS_MAP[st]?.[tic];
      const iframe = document.getElementById('report-container');
      if (filename) {{
        iframe.src = filename;
        updateStats(REPORTS_MAP[st][tic+'_stats'] || {{}});
      }} else {{
        iframe.src = "about:blank";
        alert(`找不到 ${{st}} 策略與 ${{tic}} 代號的報告，可能是數據不足或回測失敗。`);
      }}
    }}

    function fetchSymbol() {{
      const sym = document.getElementById('symbol-input').value.trim().toUpperCase();
      if (!sym) return;
      const sel = document.getElementById('ticker-select');
      if ([...sel.options].some(o => o.value === sym)) {{
        sel.value = sym;
        loadReport();
      }} else {{
        alert('該代號不在目前清單內，請確認輸入或先增加標的。');
      }}
    }}

    function updateStats(s) {{
      const set = (k, v, fix=2) => document.getElementById('st_'+k).textContent =
        (v == null ? '--' : (typeof v==='number' ? v.toFixed(fix) : v));
      set('Start', s.Start);
      set('End', s.End);
      set('Duration', s.Duration);
      set('Exposure', s['Exposure Time [%]']);
      set('EquityFinal', s['Equity Final [$]'], 2);
      set('EquityPeak', s['Equity Peak [$]'], 2);
      set('Return', s['Return [%]']);
      set('BuyHold', s['Buy & Hold Return [%]']);
      set('ReturnAnn', s['Return (Ann.) [%]']);
      set('VolAnn', s['Volatility (Ann.) [%]']);
      set('Sharpe', s['Sharpe Ratio']);
      set('Sortino', s['Sortino Ratio']);
      set('Calmar', s['Calmar Ratio']);
      set('MaxDD', s['Max. Drawdown [%]']);
      set('AvgDD', s['Avg. Drawdown [%]']);
      set('MaxDDD', s['Max. Drawdown Duration']);
      set('AvgDDD', s['Avg. Drawdown Duration']);
      set('Trades', s['# Trades'], 0);
      set('WinRate', s['Win Rate [%]']);
      set('BestTrade', s['Best Trade [%]']);
      set('WorstTrade', s['Worst Trade [%]']);
      set('AvgTrade', s['Avg. Trade [%]']);
      set('MaxTradeDur', s['Max. Trade Duration']);
      set('AvgTradeDur', s['Avg. Trade Duration']);
      set('ProfitFactor', s['Profit Factor']);
      set('Expectancy', s['Expectancy [%]']);
      set('SQN', s['SQN']);
    }}
  </script>
</body>
</html>"""

    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("Multi-strategy index.html (with query input & stats table) generated successfully.")
