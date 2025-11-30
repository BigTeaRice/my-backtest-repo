import os
import yfinance as yf
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from bokeh.plotting import output_file

# --- 1. 定義所有回測策略 ---

# RSI 策略 (超賣買入, 超買平倉)
class RsiOscillator(Strategy):
    Name = "RSI_Oscillator"
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    def init(self):
        # I() 是 backtesting.py 內建的指標計算器，會自動處理數據對齊
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)

    def next(self):
        # 平倉：RSI 穿越上限 (超買)
        if crossover(self.rsi, self.upper_bound):
            self.position.close()
        # 開倉：RSI 穿越下限 (超賣)
        elif crossover(self.lower_bound, self.rsi):
            self.buy()

# SMA 雙均線交叉策略 (快速線穿過慢速線買入)
class SmaCrossover(Strategy):
    Name = "SMA_Crossover"
    fast_period = 20
    slow_period = 50

    def init(self):
        self.sma_fast = self.I(talib.SMA, self.data.Close, self.fast_period)
        self.sma_slow = self.I(talib.SMA, self.data.Close, self.slow_period)

    def next(self):
        # 買入訊號：快線穿越慢線
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        # 賣出訊號：慢線穿越快線
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()

# MACD 訊號線交叉策略 (MACD 穿越訊號線買入)
class MacdCrossover(Strategy):
    Name = "MACD_Crossover"
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        # 計算 MACD
        self.macd, self.signal, self.hist = self.I(
            talib.MACD,
            self.data.Close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

    def next(self):
        # 買入訊號：MACD 線穿越 Signal 線
        if crossover(self.macd, self.signal):
            self.buy()
        # 賣出訊號：Signal 線穿越 MACD 線
        elif crossover(self.signal, self.macd):
            self.position.close()


# --- 2. 數據抓取 (僅使用 yfinance) ---
def get_data(ticker, start_date="2020-01-01"):
    print(f"Fetching data for {ticker} from yfinance...")
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return pd.DataFrame()

        return data
            
    except Exception as e:
        print(f"YFinance Error for {ticker}: {e}")
        return pd.DataFrame()


# --- 3. 主程式：執行回測與生成多維度報告 ---
if __name__ == "__main__":
    # 定義要回測的組合
    TICKERS = {
        "^HSI": "恆生指數 (HSI)",
        "AAPL": "Apple (AAPL)",
        "TSLA": "Tesla (TSLA)",
    }
    STRATEGIES = [RsiOscillator, SmaCrossover, MacdCrossover]
    
    OUTPUT_DIR = "public"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 用於儲存下拉選單選項的資料結構
    # { 'RSI_Oscillator': { '^HSI': 'HSI_RSI_Oscillator.html', ... }, ... }
    reports_map = {} 
    
    print("--- Starting Multi-Dimensional Backtest ---")

    for StrategyClass in STRATEGIES:
        strategy_name = StrategyClass.Name
        reports_map[strategy_name] = {}
        
        for ticker, name in TICKERS.items():
            data = get_data(ticker)
            
            # 檢查數據長度是否滿足最長指標要求 (如 SMA50)
            required_len = max(
                getattr(StrategyClass, 'rsi_window', 0), 
                getattr(StrategyClass, 'slow_period', 0),
                getattr(StrategyClass, 'signal_period', 0)
            )
            if len(data) < required_len + 50: # 額外給一點緩衝
                print(f"Skipping {ticker} / {strategy_name}: Data length {len(data)} too short.")
                continue

            # 執行回測
            try:
                bt = Backtest(data, StrategyClass, cash=100000, commission=.002)
                stats = bt.run()
                
                # 報告檔名: {策略}_{代號}.html
                safe_ticker = ticker.replace('^', '').replace('.', '_')
                report_filename = f"{strategy_name}_{safe_ticker}.html"
                output_path = os.path.join(OUTPUT_DIR, report_filename)
                
                # backtesting.py 預設輸出包含 Sharpe Ratio, Max. Drawdown 等重要指標
                bt.plot(filename=output_path, open_browser=False)
                
                reports_map[strategy_name][ticker] = report_filename
                
                print(f"✅ Success: {ticker} + {strategy_name} -> {report_filename}")

            except Exception as e:
                print(f"❌ Error during backtest for {ticker} / {strategy_name}: {e}")


    # --- 4. 生成帶有雙層選單的 index.html 主頁面 ---

    # 預設載入第一個組合的報告
    default_strategy = STRATEGIES[0].Name
    default_ticker = next(iter(TICKERS.keys()))
    default_report_file = reports_map.get(default_strategy, {}).get(default_ticker, "")

    # 生成策略下拉選單的 HTML
    strategy_options_html = ""
    for StrategyClass in STRATEGIES:
        name = StrategyClass.Name.replace('_', ' ')
        strategy_options_html += f'<option value="{StrategyClass.Name}">{name}</option>'
        
    # 生成股票代號下拉選單的 HTML
    ticker_options_html = ""
    for ticker, name in TICKERS.items():
        ticker_options_html += f'<option value="{ticker}">{name} ({ticker})</option>'

    # 將 reports_map 轉換為 JavaScript 字典，用於前端查詢檔案名
    import json
    reports_json = json.dumps(reports_map)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>多策略回測報告中心</title>
        <style>
            body {{ font-family: 'Inter', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .header-control {{ margin-bottom: 25px; padding: 15px; border-bottom: 2px solid #e0e0e0; }}
            h2 {{ color: #333; margin-top: 0; font-weight: 600; }}
            label {{ font-weight: bold; margin-right: 10px; color: #555; }}
            select {{ padding: 10px 15px; font-size: 16px; border: 1px solid #ccc; border-radius: 8px; margin-right: 20px; background-color: #fff; cursor: pointer; transition: border-color 0.3s; }}
            select:hover {{ border-color: #007bff; }}
            #report-container {{ width: 100%; height: calc(100vh - 200px); border: none; border-radius: 8px; margin-top: 20px; background-color: #fff; box-shadow: inset 0 0 5px rgba(0,0,0,0.05); }}
            .note {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-control">
                <h2>多策略回測報告中心</h2>
                <label for="strategy-select">選擇策略：</label>
                <select id="strategy-select" onchange="loadReport()">
                    {strategy_options_html}
                </select>

                <label for="ticker-select">選擇股票代號：</label>
                <select id="ticker-select" onchange="loadReport()">
                    {ticker_options_html}
                </select>
                
                <p class="note">報告包含夏普指數 (Sharpe Ratio)、最大回撤 (Max. Drawdown) 等關鍵指標。</p>
                <p class="note">資料來源: yfinance / 報告由 GitHub Actions 自動生成</p>
            </div>
            
            <!-- 使用 iframe 載入回測報告 -->
            <iframe id="report-container" src="{default_report_file}" frameborder="0"></iframe>

            <script>
                // 從 Python 導入的報告對應表
                const REPORTS_MAP = {reports_json};
                
                // 網頁載入後立即執行，確保選單和 iframe 同步
                window.onload = function() {{
                    loadReport();
                }};

                function loadReport() {{
                    const strategySelect = document.getElementById('strategy-select');
                    const tickerSelect = document.getElementById('ticker-select');
                    const reportContainer = document.getElementById('report-container');

                    const selectedStrategy = strategySelect.value;
                    const selectedTicker = tickerSelect.value;
                    
                    // 從 REPORTS_MAP 中查找對應的檔名
                    const filename = REPORTS_MAP[selectedStrategy] ? REPORTS_MAP[selectedStrategy][selectedTicker] : null;

                    if (filename) {{
                        reportContainer.src = filename;
                    }} else {{
                        // 如果組合不存在 (例如某代號數據不足)
                        reportContainer.src = "about:blank"; // 顯示空白頁
                        alert(`找不到 ${selectedStrategy} 策略與 ${selectedTicker} 代號的報告，可能是數據不足或回測失敗。`);
                    }}
                }}
            </script>
        </div>
    </body>
    </html>
    """
    
    # 寫入 index.html
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print("Multi-strategy index.html generated successfully.")
