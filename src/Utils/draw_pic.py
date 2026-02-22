"""
画图插件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class QuantTearSheet:
    @staticmethod
    def calculate_drawdown(equity_curve):
        """计算动态回撤序列"""
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown

    @staticmethod
    def plot_comparison(price_series, strategies_dict, focus_trade_strategy=None):
        """
        标准化的策略对比绘图引擎

        参数:
        price_series (pd.Series): 标的资产的价格序列
        strategies_dict (dict): 策略字典，格式为 {'Strategy Name': {'equity': pd.Series, 'trades': pd.DataFrame}}
        focus_trade_strategy (str): 需要在图表底部展示具体买卖点位的策略名称
        """
        # 设置画布布局：3个子图，比例为 2:1:1.5
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1.5]})
        fig.suptitle('Quant Strategy Tear Sheet: Performance & Execution Analysis', fontsize=16, fontweight='bold')
        plt.style.use('bmh')  # 使用专业图表风格

        # ==========================================
        # Panel 1: 绝对净值对比 (Equity Curves)
        # ==========================================
        ax1 = axes[0]
        # 绘制基准 (Buy & Hold) - 假设全仓买入
        benchmark = (price_series / price_series.iloc[0]) * \
                    strategies_dict[list(strategies_dict.keys())[0]]['equity'].iloc[0]
        ax1.plot(benchmark, label='Benchmark (Buy & Hold)', color='grey', linestyle='--', alpha=0.7)

        colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
        for i, (name, data) in enumerate(strategies_dict.items()):
            ax1.plot(data['equity'], label=name, color=colors[i % len(colors)], linewidth=2)

        ax1.set_title('Cumulative Portfolio Value', fontsize=12, loc='left')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # ==========================================
        # Panel 2: 动态回撤 (Underwater Plot)
        # ==========================================
        ax2 = axes[1]
        ax2.fill_between(benchmark.index, QuantTearSheet.calculate_drawdown(benchmark), 0,
                         color='grey', alpha=0.2, label='Benchmark Drawdown')

        for i, (name, data) in enumerate(strategies_dict.items()):
            dd = QuantTearSheet.calculate_drawdown(data['equity'])
            ax2.plot(dd, label=f'{name} DD', color=colors[i % len(colors)], linewidth=1.5)

        ax2.set_title('Underwater Plot (Drawdown Analysis)', fontsize=12, loc='left')
        ax2.set_ylabel('Drawdown')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        # ==========================================
        # Panel 3: 微观交易结构 (Trade Execution)
        # ==========================================
        ax3 = axes[2]
        ax3.plot(price_series, label='Underlying Price', color='black', alpha=0.6, linewidth=1)

        if focus_trade_strategy and focus_trade_strategy in strategies_dict:
            trades = strategies_dict[focus_trade_strategy]['trades']
            if not trades.empty:
                buys = trades[trades['action'] == 'Buy']
                sells = trades[trades['action'] == 'Sell']

                ax3.scatter(buys['index'], buys['price'], marker='^', color='green', s=60,
                            label=f'Buy ({len(buys)})', zorder=5)
                ax3.scatter(sells['index'], sells['price'], marker='v', color='red', s=60,
                            label=f'Sell ({len(sells)})', zorder=5)

                # 如果有均线数据(如条件网格)，则绘制
                if 'ma' in trades.columns:
                    ax3.plot(trades['index'], trades['ma'], color='orange', linestyle=':',
                             label='Trend Filter (MA)', alpha=0.8)

            ax3.set_title(f'Trade Execution Diagram: {focus_trade_strategy}', fontsize=12, loc='left')
        else:
            ax3.set_title('Trade Execution Diagram (No strategy selected)', fontsize=12, loc='left')

        ax3.set_ylabel('Price')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"../pic/Trade Execution Diagram-{focus_trade_strategy}.png")
        plt.show()


# ==========================================
# 示例：如何向标准化引擎喂入数据
# ==========================================
"""
假设您已经运行了前面的策略，获得了以下变量：
1. df_market['Close'] (价格序列 pd.Series)
2. curve_neutral, trades_neutral (中性网格的回测结果)
3. curve_cond, trades_cond (条件网格的回测结果)

您只需要按照以下字典格式组织数据并调用：

strategies_results = {
    'Neutral Grid': {
        'equity': curve_neutral['Portfolio_Value'],
        'trades': trades_neutral
    },
    'Conditional Grid (MA+ATR)': {
        'equity': curve_cond['Portfolio_Value'],
        'trades': trades_cond
    }
}

# 调用标准化绘图方法，并将微观审查焦点放在"条件网格"上
QuantTearSheet.plot_comparison(
    price_series=df_market['Close'], 
    strategies_dict=strategies_results, 
    focus_trade_strategy='Conditional Grid (MA+ATR)'
)
"""