"""
趋势VS震荡的两分状态机+动态网格(ATR)的策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.Utils.draw_pic import QuantTearSheet


# ==========================================
# 1. 数据生成模块 (包含 OHLC)
# ==========================================
def generate_ohlc_data(days=250, start_price=1.0):
    np.random.seed(42)
    # 模拟基础收盘价：包含一个深度的"U型"反转，前120天单边下跌，后130天震荡上涨
    t = np.linspace(0, 10, days)
    trend = -np.sin(t * 0.5) * 0.3  # 制造单边趋势
    noise = np.random.normal(0, 0.02, days)

    close_prices = start_price + trend + np.cumsum(noise)
    close_prices = np.maximum(close_prices, 0.1)  # 防止价格为负

    df = pd.DataFrame({'Close': close_prices})
    # 模拟真实市场的高低价，用于计算 ATR
    df['High'] = df['Close'] * (1 + np.abs(np.random.normal(0, 0.015, days)))
    df['Low'] = df['Close'] * (1 - np.abs(np.random.normal(0, 0.015, days)))

    # 模拟前收盘价
    df['Prev_Close'] = df['Close'].shift(1)
    return df.dropna().reset_index(drop=True)


# ==========================================
# 2. 核心策略引擎：动态条件网格
# ==========================================
class ConditionalGridStrategy:
    def __init__(self, df, initial_capital=100000, trade_amount=5000,
                 atr_period=14, atr_multiplier=1.5, ma_period=20):
        self.df = df
        self.cash = initial_capital * 0.5
        self.holdings = (initial_capital * 0.5) / df.loc[0, 'Close']
        self.trade_amount = trade_amount
        self.atr_multi = atr_multiplier

        self.last_trade_price = df.loc[0, 'Close']
        self.history = []
        self.trades = []

        # 计算技术指标
        self._calculate_indicators(atr_period, ma_period)

    def _calculate_indicators(self, atr_period, ma_period):
        # 1. 计算均线 (趋势过滤器)
        self.df['MA'] = self.df['Close'].rolling(window=ma_period).mean()

        # 2. 计算 ATR (波动率步长)
        tr1 = self.df['High'] - self.df['Low']
        tr2 = np.abs(self.df['High'] - self.df['Prev_Close'])
        tr3 = np.abs(self.df['Low'] - self.df['Prev_Close'])
        self.df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = self.df['TR'].rolling(window=atr_period).mean()

        # 避免早期数据缺失
        self.df.bfill(inplace=True)

    def run_backtest(self):
        for i in range(len(self.df)):
            current_price = self.df.loc[i, 'Close']
            current_ma = self.df.loc[i, 'MA']
            current_atr = self.df.loc[i, 'ATR']

            # 动态网格步长 (当前价格的百分比)
            dynamic_grid_size = (current_atr * self.atr_multi) / current_price

            action = None

            # --- 卖出逻辑 (无趋势限制，反弹即卖) ---
            if current_price >= self.last_trade_price * (1 + dynamic_grid_size):
                if self.holdings * current_price >= self.trade_amount:
                    sell_shares = self.trade_amount / current_price
                    self.cash += self.trade_amount
                    self.holdings -= sell_shares
                    self.last_trade_price = current_price
                    action = 'Sell'

            # --- 买入逻辑 (叠加趋势过滤器！) ---
            elif current_price <= self.last_trade_price * (1 - dynamic_grid_size):
                # 核心改变：只有价格在 MA 之上(或乖离不严重)，才允许买入。拒绝单边下跌接飞刀。
                is_uptrend_or_震荡 = current_price > current_ma

                if is_uptrend_or_震荡 and self.cash >= self.trade_amount:
                    buy_shares = self.trade_amount / current_price
                    self.cash -= self.trade_amount
                    self.holdings += buy_shares
                    self.last_trade_price = current_price
                    action = 'Buy'

            # 记录净值
            total_value = self.cash + self.holdings * current_price
            self.history.append(total_value)

            if action:
                self.trades.append({
                    'index': i, 'price': current_price, 'action': action, 'ma': current_ma
                })

        return pd.DataFrame(self.history, columns=['Portfolio_Value']), pd.DataFrame(self.trades)


# ==========================================
# 3. 运行与可视化
# ==========================================
df_market = generate_ohlc_data(days=250)

# 初始化策略：10万本金，每次交易5000，使用 1.5倍ATR作为网格，20日均线作为防线
strategy = ConditionalGridStrategy(df_market, atr_multiplier=1.5, ma_period=20)
curve, trades = strategy.run_backtest()

print(f"回测完成。总交易次数: {len(trades)} 次")
# (此处省略 matplotlib 绘图代码，与上一版本类似，但会在买卖点上展现出明显的"下跌熔断"特征)

# ==========================================
# 4. 绘图展示
# ==========================================
strategies_results = {
    # 'Neutral Grid': {
    #     'equity': curve_neutral['Portfolio_Value'],
    #     'trades': trades_neutral
    # },
    'Conditional Grid (MA+ATR)': {
        'equity': curve['Portfolio_Value'],
        'trades': trades
    }
}

# 调用标准化绘图方法，并将微观审查焦点放在"条件网格"上
QuantTearSheet.plot_comparison(
    price_series=df_market['Close'],
    strategies_dict=strategies_results,
    focus_trade_strategy='Conditional Grid (MA+ATR)'
)