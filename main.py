import os
import yfinance as yf
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from bokeh.plotting import output_file

# --- 策略定義 (RSI Strategy, 參考您的 IPYNB) ---
class RsiOscillator(Strategy):
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    def init(self):
        # 使用 TA-Lib 計算 RSI
        # 注意: backtesting.py 傳入的是 numpy array，talib 可以直接計算
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)

    def next(self):
        if crossover(self.rsi, self.upper_bound):
            self.position.close()
        elif crossover(self.lower_bound, self.rsi):
            self.buy()

# --- 數據抓取 (僅使用 yfinance) ---
def get_data(ticker, start_date="2020-01-01"):
    print(f"Fetching data for {ticker} from yfinance...")
    
    try:
        # 下載數據
        data = yf.download(ticker, start=start_date, progress=False)
        
        # --- 數據清洗與格式處理 ---
        # 1. 處理 yfinance 新版的 MultiIndex Columns (例如: ('Close', '^HSI') -> 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # 2. 移除時區資訊 (Backtesting.py 有時對時區敏感)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # 3. 確保包含 OHLCV 欄位且不為空
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            print(f"Error: Missing required columns. Available: {data.columns}")
            return pd.DataFrame()

        if len(data) > 0:
            print(f"Successfully fetched {len(data)} rows.")
            return data
            
    except Exception as e:
        print(f"YFinance Error: {e}")
    
    return pd.DataFrame()

# --- 主程式 ---
if __name__ == "__main__":
    # 設定回測目標
    TICKER = "^HSI"  # 恆生指數
    
    # 建立輸出資料夾 (GitHub Pages 需要此 public 資料夾)
    OUTPUT_DIR = "public"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 獲取數據
    data = get_data(TICKER)

    if not data.empty:
        # 執行回測 (初始資金 100,000, 手續費 0.2%)
        bt = Backtest(data, RsiOscillator, cash=100000, commission=.002)
        stats = bt.run()
        print(stats)

        # 生成 HTML 報表
        # 將檔名設為 index.html，這樣 GitHub Pages 首頁就能直接顯示
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        
        output_file(output_path)
        bt.plot(filename=output_path, open_browser=False)
        
        print(f"Report generated successfully at: {output_path}")
    else:
        print("No data found. Report generation skipped.")
