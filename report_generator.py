"""
报告生成模块
生成HTML、JSON等格式的回测报告
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os


class ReportGenerator:
    """报告生成器"""
    
    def generate_report(self,
                       strategy_name: str,
                       symbol: str,
                       start_date: str,
                       end_date: str,
                       initial_capital: float,
                       performance_stats: Dict,
                       trades: Optional[pd.DataFrame] = None,
                       equity_curve: Optional[pd.Series] = None) -> Dict:
        """
        生成完整回测报告
        
        Returns:
            报告数据字典
        """
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'strategy': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital
            },
            'performance_stats': performance_stats,
            'trades': self._format_trades(trades) if trades is not None else [],
            'charts': self._generate_charts_data(performance_stats, equity_curve),
            'analysis': self._generate_analysis(performance_stats)
        }
        
        return report
    
    def _format_trades(self, trades: pd.DataFrame) -> List[Dict]:
        """格式化交易记录"""
        if trades is None or trades.empty:
            return []
        
        formatted_trades = []
        for _, trade in trades.iterrows():
            formatted_trades.append({
                'entry_time': str(trade.get('EntryTime', '')),
                'exit_time': str(trade.get('ExitTime', '')),
                'size': trade.get('Size', 0),
                'entry_price': trade.get('EntryPrice', 0),
                'exit_price': trade.get('ExitPrice', 0),
                'pnl': trade.get('PnL', 0),
                'pnl_pct': trade.get('ReturnPct', 0),
                'duration': str(trade.get('Duration', ''))
            })
        
        return formatted_trades
    
    def _generate_charts_data(self, 
                            performance_stats: Dict, 
                            equity_curve: pd.Series) -> Dict:
        """生成图表数据"""
        charts = {}
        
        # 这里可以添加图表生成逻辑
        # 例如：equity_curve.to_dict() if equity_curve is not None else {}
        
        return charts
    
    def _generate_analysis(self, performance_stats: Dict) -> Dict:
        """生成分析报告"""
        analysis = {
            'rating': self._rate_performance(performance_stats),
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # 根据指标评估
        sharpe = performance_stats.get('sharpe_ratio', 0)
        max_dd = performance_stats.get('max_drawdown', 0)
        win_rate = performance_stats.get('win_rate', 0)
        
        # 评估夏普比率
        if sharpe > 1.5:
            analysis['strengths'].append('优秀的风险调整后收益')
        elif sharpe < 0.5:
            analysis['weaknesses'].append('风险调整后收益较低')
        
        # 评估最大回撤
        if max_dd < 0.1:
            analysis['strengths'].append('回撤控制良好')
        elif max_dd > 0.3:
            analysis['weaknesses'].append('回撤较大，风险较高')
        
        # 评估胜率
        if win_rate > 0.6:
            analysis['strengths'].append('交易胜率较高')
        elif win_rate < 0.4:
            analysis['weaknesses'].append('交易胜率较低')
        
        # 生成建议
        if sharpe < 1.0:
            analysis['recommendations'].append('考虑优化策略参数以改善风险收益比')
        if max_dd > 0.2:
            analysis['recommendations'].append('建议添加止损机制控制回撤')
        if performance_stats.get('total_trades', 0) < 10:
            analysis['recommendations'].append('交易次数较少，可能需要更长的回测周期')
        
        return analysis
    
    def _rate_performance(self, performance_stats: Dict) -> str:
        """评级性能"""
        score = 0
        
        # 夏普比率评分
        sharpe = performance_stats.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            score += 3
        elif sharpe > 1.0:
            score += 2
        elif sharpe > 0.5:
            score += 1
        
        # 最大回撤评分
        max_dd = performance_stats.get('max_drawdown', 0)
        if max_dd < 0.1:
            score += 3
        elif max_dd < 0.2:
            score += 2
        elif max_dd < 0.3:
            score += 1
        
        # 胜率评分
        win_rate = performance_stats.get('win_rate', 0)
        if win_rate > 0.6:
            score += 2
        elif win_rate > 0.5:
            score += 1
        
        # 总收益率评分
        total_return = performance_stats.get('total_return', 0)
        if total_return > 0.5:
            score += 3
        elif total_return > 0.2:
            score += 2
        elif total_return > 0:
            score += 1
        
        # 根据总分评级
        if score >= 8:
            return '优秀 (A)'
        elif score >= 6:
            return '良好 (B)'
        elif score >= 4:
            return '一般 (C)'
        else:
            return '较差 (D)'
    
    def save_html_report(self, report_data: Dict, filepath: str):
        """保存HTML报告"""
        html_template = self._create_html_template(report_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_template)
    
    def _create_html_template(self, report_data: Dict) -> str:
        """创建HTML模板"""
        stats = report_data['performance_stats']
        metadata = report_data['metadata']
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告 - {metadata['strategy']} - {metadata['symbol']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; }}
        .header h1 {{ color: #333; margin-bottom: 10px; }}
        .header .subtitle {{ color: #666; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #333; margin: 10px 0; }}
        .metric-card .label {{ color: #666; font-size: 14px; }}
        .positive {{ color: #4CAF50 !important; }}
        .negative {{ color: #f44336 !important; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .section {{ margin: 40px 0; }}
        .section-title {{ font-size: 20px; color: #333; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
        .analysis {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 40px; color: #888; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 回测报告</h1>
            <div class="subtitle">
                <strong>策略:</strong> {metadata['strategy']} | 
                <strong>标的:</strong> {metadata['symbol']} | 
                <strong>期间:</strong> {metadata['start_date']} 至 {metadata['end_date']}
            </div>
            <div class="subtitle">
                生成时间: {metadata['generated_at']}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📈 关键指标</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">总收益率</div>
                    <div class="value {'positive' if stats.get('total_return', 0) > 0 else 'negative'}">
                        {stats.get('total_return', 0):.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">年化收益率</div>
                    <div class="value {'positive' if stats.get('annual_return', 0) > 0 else 'negative'}">
                        {stats.get('annual_return', 0):.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">夏普比率</div>
                    <div class="value">
                        {stats.get('sharpe_ratio', 0):.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">最大回撤</div>
                    <div class="value negative">
                        {stats.get('max_drawdown', 0):.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">胜率</div>
                    <div class="value">
                        {stats.get('win_rate', 0):.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">总交易次数</div>
                    <div class="value">
                        {stats.get('total_trades', 0):,d}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">💰 资金曲线</h2>
            <div style="background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center;">
                <p>净值曲线图表 (需要JavaScript支持)</p>
                <p><small>在实际系统中，这里会显示Plotly或Matplotlib生成的图表</small></p>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📋 详细统计</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>初始资金</td>
                    <td>¥{stats.get('initial_capital', 0):,.2f}</td>
                    <td>回测开始资金</td>
                </tr>
                <tr>
                    <td>最终净值</td>
                    <td>¥{stats.get('final_value', 0):,.2f}</td>
                    <td>回测结束总资产</td>
                </tr>
                <tr>
                    <td>年化波动率</td>
                    <td>{stats.get('volatility', 0):.2%}</td>
                    <td>价格波动程度</td>
                </tr>
                <tr>
                    <td>盈亏因子</td>
                    <td>{stats.get('profit_factor', 0):.2f}</td>
                    <td>盈利与亏损比例</td>
                </tr>
                <tr>
                    <td>索提诺比率</td>
                    <td>{stats.get('sortino_ratio', 0):.2f}</td>
                    <td>下行风险调整收益</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2 class="section-title">🔍 策略分析</h2>
            <div class="analysis">
                <h3>绩效评级: {report_data['analysis']['rating']}</h3>
                <p><strong>优势:</strong> {'，'.join(report_data['analysis']['strengths']) if report_data['analysis']['strengths'] else '无明显优势'}</p>
                <p><strong>不足:</strong> {'，'.join(report_data['analysis']['weaknesses']) if report_data['analysis']['weaknesses'] else '无明显不足'}</p>
                <p><strong>改进建议:</strong> {'；'.join(report_data['analysis']['recommendations']) if report_data['analysis']['recommendations'] else '继续保持当前策略'}</p>
            </div>
        </div>
        
        <div class="footer">
            <p>本报告由多策略回测系统自动生成 | 投资有风险，决策需谨慎</p>
            <p>报告时间: {metadata['generated_at']}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
