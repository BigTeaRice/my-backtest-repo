#!/usr/bin/env python3
# main.py
import os
import json
import yfinance as yf
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from bokeh.plotting import output_file   # backtesting.py 绘图依赖

# ------------------------------------------------------------------
# 1. 策略定义
# ------------------------------------------------------------------
class RsiOscillator(Strategy):
    Name = "RSI_Oscillator"
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)

    def next(self):
        if crossover(self.rsi, self.upper_bound):
            self.position.close()
        elif crossover(self.lower_bound, self.rsi):
            self.buy()


class SmaCrossover(Strategy):
    Name = "SMA_Crossover"
    fast_period = 20
    slow_period = 50

    def init(self):
        self.sma_fast = self.I(talib.SMA, self.data.Close, self.fast_period)
        self.sma_slow = self.I(talib.SMA, self.data.Close, self.slow_period)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()


class MacdCrossover(Strategy):
    Name = "MACD_Crossover"
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        self.macd, self.signal, _ = self.I(
            talib.MACD,
            self.data.Close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

    def next(self):
        if crossover(self.macd, self.signal):
            self.buy()
        elif crossover(self.signal, self.macd):
            self.position.close()


# ------------------------------------------------------------------
# 2. 数据抓取
# ------------------------------------------------------------------
def get_data(ticker: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    print(f"Fetching data for {ticker} from yfinance...")
    try:
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required):
            return pd.DataFrame()

        return data
    except Exception as e:
        print(f"YFinance Error for {ticker}: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------------
# 3. 主程序
# ------------------------------------------------------------------
if __name__ == "__main__":
    TICKERS = {
        "^HSI": "恆生指數 (HSI)",
        "AAPL": "Apple (AAPL)",
        "TSLA": "Tesla (TSLA)",
    }
    STRATEGIES = [RsiOscillator, SmaCrossover, MacdCrossover]

    OUTPUT_DIR = "public"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reports_map = {}   # {策略: {代號: 文件名}}

    print("--- Starting Multi-Dimensional Backtest ---")

    for StrategyCls in STRATEGIES:
        strategy_name = StrategyCls.Name
        reports_map[strategy_name] = {}

        for ticker, desc in TICKERS.items():
            data = get_data(ticker)
            required_len = max(
                getattr(StrategyCls, "rsi_window", 0),
                getattr(StrategyCls, "slow_period", 0),
                getattr(StrategyCls, "signal_period", 0)
            )
            if len(data) < required_len + 50:
                print(f"Skipping {ticker} / {strategy_name}: Data too short.")
                continue

            try:
                bt = Backtest(data, StrategyCls, cash=100_000, commission=.002)
                stats = bt.run()

                safe_ticker = ticker.replace("^", "").replace(".", "_")
                report_filename = f"{strategy_name}_{safe_ticker}.html"
                output_path = os.path.join(OUTPUT_DIR, report_filename)
                bt.plot(filename=output_path, open_browser=False)

                reports_map[strategy_name][ticker] = report_filename
                print(f"✅ Success: {ticker} + {strategy_name} -> {report_filename}")

            except Exception as e:
                print(f"❌ Error during backtest for {ticker} / {strategy_name}: {e}")

    # ------------------------------------------------------------------
    # 4. 生成 index.html
    # ------------------------------------------------------------------
    default_strategy = STRATEGIES[0].Name
    default_ticker = next(iter(TICKERS.keys()))
    default_report = reports_map.get(default_strategy, {}).get(default_ticker, "")

    strategy_options = "\n".join(
        f'<option value="{s.Name}">{s.Name.replace("_", " ")}</option>'
        for s in STRATEGIES
    )
    ticker_options = "\n".join(
        f'<option value="{k}">{v} ({k})</option>'
        for k, v in TICKERS.items()
    )

    reports_json = json.dumps(reports_map)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多策略回測報告中心</title>
    <style>
        body{{font-family:'Inter',Arial,sans-serif;margin:0;padding:20px;background:#f4f7f9;}}
        .container{{max-width:1200px;margin:0 auto;background:white;padding:20px;border-radius:12px;box-shadow:0 4px 8px rgba(0,0,0,0.1);}}
        .header-control{{margin-bottom:25px;padding:15px;border-bottom:2px solid #e0e0e0;}}
        h2{{color:#333;margin-top:0;font-weight:600;}}
        label{{font-weight:bold;margin-right:10px;color:#555;}}
        select{{padding:10px 15px;font-size:16px;border:1px solid #ccc;border-radius:8px;margin-right:20px;background:white;cursor:pointer;transition:border-color .3s;}}
        select:hover{{border-color:#007bff;}}
        #report-container{{width:100%;height:calc(100vh - 200px);border:none;border-radius:8px;margin-top:20px;background:white;box-shadow:inset 0 0 5px rgba(0,0,0,0.05);}}
        .note{{color:#666;font-size:0.9em;margin-top:10px;}}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-control">
            <h2>多策略回測報告中心</h2>
            <label for="strategy-select">選擇策略：</label>
            <select id="strategy-select" onchange="loadReport()">{strategy_options}</select>

            <label for="ticker-select">選擇股票代號：</label>
            <select id="ticker-select" onchange="loadReport()">{ticker_options}</select>

            <p class="note">報告包含夏普指數 (Sharpe Ratio)、最大回撤 (Max. Drawdown) 等關鍵指標。</p>
            <p class="note">資料來源：yfinance / 報告由 GitHub Actions 自動生成</p>
        </div>

        <iframe id="report-container" src="{default_report}" frameborder="0"></iframe>

        <script>
            const REPORTS_MAP = {reports_json};

            window.onload = () => loadReport();

            function loadReport() {{
                const strategy = document.getElementById('strategy-select').value;
                const ticker   = document.getElementById('ticker-select').value;
                const filename = REPORTS_MAP[strategy]?.[ticker];
                const iframe   = document.getElementById('report-container');

                if (filename) {{
                    iframe.src = filename;
                }} else {{
                    iframe.src = "about:blank";
                    alert(`找不到 ${{strategy}} 策略與 ${{ticker}} 代號的報告，可能是數據不足或回測失敗。`);
                }}
            }}
        </script>
    </div>
</body>
</html>"""

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Multi-strategy index.html generated successfully.")
