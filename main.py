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
        # 使用 TA-Lib 計算 RSI
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)

    def next(self):
        if crossover(self.rsi, self.upper_bound):
            self.position.close()
        elif crossover(self.lower_bound, self.rsi):
            self.buy()

# --- 數據抓取 ---
def get_data(ticker, start_date="2020-01-01"):
    print(f"Fetching data for {ticker}...")
    
    # 優先使用 yfinance
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        
        # 處理 MultiIndex Column (yfinance 新版特性)
        if isinstance(data.columns, pd.MultiIndex):
             data.columns = [c[0] for c in data.columns]
        
        # 確保數據格式正確
        if len(data) > 0:
            return data
    except Exception as e:
        print(f"YFinance Error: {e}")

    # 若 yfinance 失敗，可在此處加入 Alpha Vantage 邏輯 (需 API Key)
    # key = os.environ.get('ALPHAVANTAGE_API_KEY')
    # ... Alpha Vantage code ...
    
    return pd.DataFrame()

# --- 主程式 ---
if __name__ == "__main__":
    # 設定回測目標
    TICKER = "^HSI"  # 恆生指數
    
    # 建立輸出資料夾 (GitHub Pages 需要)
    OUTPUT_DIR = "public"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data = get_data(TICKER)

    if not data.empty:
        # 執行回測
        bt = Backtest(data, RsiOscillator, cash=100000, commission=.002)
        stats = bt.run()
        print(stats)

        # 生成 HTML (存入 public/index.html)
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        output_file(output_path)
        bt.plot(filename=output_path, open_browser=False)
        print(f"Report generated successfully at {output_path}")
    else:
        print("No data found. Report generation skipped.")
