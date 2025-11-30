import os
import yfinance as yf
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from bokeh.plotting import output_file

# --- 策略定義 (RSI Strategy) ---
class RsiOscillator(Strategy):
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

# --- 數據抓取 (僅使用 yfinance) ---
def get_data(ticker, start_date="2020-01-01"):
    print(f"Fetching data for {ticker}...")
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

# --- 主程式 ---
if __name__ == "__main__":
    # 定義要回測的股票代號清單
    TICKERS = {
        "^HSI": "Hang Seng Index (HSI)",
        "AAPL": "Apple Inc. (AAPL)",
        "GOOG": "Alphabet Inc. (GOOG)",
        "TSLA": "Tesla (TSLA)",
        "0005.HK": "滙豐控股 (HSBC HK)",
    }
    
    OUTPUT_DIR = "public"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 用於生成下拉選單的 HTML 內容
    dropdown_options = ""
    
    # --- 1. 遍歷所有代號並生成獨立報告 ---
    for ticker, name in TICKERS.items():
        data = get_data(ticker)
        
        # 檢查是否有足夠數據進行回測
        if len(data) < RsiOscillator.rsi_window:
            print(f"Skipping {ticker}: Not enough data.")
            continue
            
        # 執行回測
        bt = Backtest(data, RsiOscillator, cash=100000, commission=.002)
        stats = bt.run()
        print(f"\n--- {ticker} Backtest Stats ---\n{stats}")

        # 生成報告 (檔名為 Ticker.html)
        report_filename = f"{ticker.replace('^', '').replace('.', '_')}.html"
        output_path = os.path.join(OUTPUT_DIR, report_filename)
        
        # 使用 backtesting.py 的 plot 方法生成報告
        bt.plot(filename=output_path, open_browser=False)
        print(f"Report generated: {output_path}")
        
        # 準備下拉選單的 HTML 標籤
        dropdown_options += f'<option value="{report_filename}">{ticker} - {name}</option>'

    # --- 2. 生成帶有下拉選單的 index.html 主頁面 ---
    
    # 預設載入第一個代號的報告
    default_report = next(iter(TICKERS.keys())).replace('^', '').replace('.', '_') + '.html'
    
    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>RSI 回測結果</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            #report-container {{ width: 100%; height: 100vh; border: none; }}
            .header-control {{ margin-bottom: 20px; }}
            select {{ padding: 10px; font-size: 16px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header-control">
            <h2>RSI 策略回測報告</h2>
            <label for="ticker-select">選擇股票代號：</label>
            <select id="ticker-select" onchange="loadReport(this.value)">
                {dropdown_options}
            </select>
            <p>資料來源: yfinance / 報告由 GitHub Actions 自動生成</p>
        </div>
        
        <iframe id="report-container" src="{default_report}" frameborder="0"></iframe>

        <script>
            function loadReport(filename) {{
                document.getElementById('report-container').src = filename;
            }}
        </script>
    </body>
    </html>
    """
    
    # 寫入 index.html
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print("index.html with dropdown selector generated successfully.")
