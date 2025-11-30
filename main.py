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
