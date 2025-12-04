# config.py
from datetime import datetime

STOCKS_CONFIG = {
    "港股": {
        "^HSI": "恒生指数",
        "0700.HK": "腾讯控股",
        "9988.HK": "阿里巴巴-SW",
        "3690.HK": "美团-W",
        "1810.HK": "小米集团-W",
    },
    "美股": {
        "SPY": "标普500 ETF",
        "QQQ": "纳指100 ETF",
        "AAPL": "苹果",
        "MSFT": "微软",
        "GOOGL": "谷歌",
        "AMZN": "亚马逊",
        "TSLA": "特斯拉",
        "NVDA": "英伟达",
    },
    "A股(美股ADR)": {
        "BABA": "阿里巴巴",
        "JD": "京东",
        "PDD": "拼多多",
        "NIO": "蔚来",
    }
}

BACKTEST_CONFIG = {
    "start_date": "2020-01-01",
    "end_date": datetime.now().strftime("%Y-%m-%d"),
    "initial_cash": 100000,
    "commission": 0.002,
    "slippage": 0.001,
}

STRATEGY_PARAMS = {
    "RSI": {"upper": 70, "lower": 30, "window": 14},
    "SMA": {"fast": 20, "slow": 50},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BB": {"window": 20, "dev": 2.0},
    "Stoch": {"k_period": 14, "d_period": 3, "smooth_k": 3}
}

ANALYSIS_CONFIG = {
    "risk_free_rate": 0.02,
    "benchmark": "^GSPC",
    "max_drawdown_limit": 0.2,
    "min_sharpe_ratio": 1.0,
}
