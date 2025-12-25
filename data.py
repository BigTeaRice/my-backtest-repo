# data.py – Polygon 数据源
from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "AKxxxxxxxxxx"   # ← 换成你的 Key
client = RESTClient(API_KEY)

def polygon_daily(ticker, start, end):
    """返回与 yfinance 格式一致的 DataFrame"""
    resp = client.get_aggs(ticker, 1, "day", start, end)
    if not resp:
        return None
    df = pd.DataFrame([{
        "Open":  r.open,
        "High":  r.high,
        "Low":   r.low,
        "Close": r.close,
        "Volume": r.volume,
        "Date":  pd.to_datetime(r.timestamp, unit="ms")
    } for r in resp])
    df.set_index("Date", inplace=True)
    return df
