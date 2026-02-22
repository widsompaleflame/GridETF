"""
# 中性网格策略 (Neutral Grid Strategy)

## 策略逻辑：
- 底仓构建： 起始资金 50% 买入 ETF，50% 持有现金（应对下跌补仓）。
- 网格生成： 以当前价格为中枢，每跌 X% 买入一格，每涨 X% 卖出一格。
- 资金管理： 等额下单（每次买卖固定金额，如 10000 元）。

我们将模拟一个 “高波动震荡” 的市场环境（最适合网格），来验证策略收益。

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 核心策略类：网格交易 (Grid Trading Class)
# ==========================================
class GridStrategy:
    def __init__(self, price_data, initial_capital=100000, grid_size=0.03, trade_amount=5000):
        """
        price_data: 价格序列
        initial_capital: 初始总资金 (e.g., 10万)
        grid_size: 网格密度 (e.g., 0.03 = 3%)
        trade_amount: 每次加减仓金额 (e.g., 5000元)
        """
        self.prices = price_data
        self.cash = initial_capital * 0.5  # 50% 现金
        self.holdings = (initial_capital * 0.5) / price_data[0]  # 50% 建仓
        self.grid_size = grid_size
        self.trade_amount = trade_amount

        # 记录基准价格 (上一次成交价)
        self.last_trade_price = price_data[0]

        # 记录账户价值
        self.history = []
        self.trades = []  # 记录交易点

    def run_backtest(self):
        for i, current_price in enumerate(self.prices):
            action = None

            # --- 卖出逻辑 (Sell Logic) ---
            # 当前价格 > 上次成交价 * (1 + 网格大小) -> 止盈一格
            if current_price >= self.last_trade_price * (1 + self.grid_size):
                if self.holdings * current_price >= self.trade_amount:  # 确保持仓足够
                    sell_shares = self.trade_amount / current_price
                    self.cash += self.trade_amount
                    self.holdings -= sell_shares
                    self.last_trade_price = current_price  # 更新基准价
                    action = 'Sell'

            # --- 买入逻辑 (Buy Logic) ---
            # 当前价格 < 上次成交价 * (1 - 网格大小) -> 补仓一格
            elif current_price <= self.last_trade_price * (1 - self.grid_size):
                if self.cash >= self.trade_amount:  # 确保现金足够
                    buy_shares = self.trade_amount / current_price
                    self.cash -= self.trade_amount
                    self.holdings += buy_shares
                    self.last_trade_price = current_price  # 更新基准价
                    action = 'Buy'

            # --- 记录状态 ---
            total_value = self.cash + self.holdings * current_price
            self.history.append(total_value)

            if action:
                self.trades.append({
                    'index': i,
                    'price': current_price,
                    'action': action,
                    'value': total_value
                })

        return pd.DataFrame(self.history, columns=['Portfolio_Value']), pd.DataFrame(self.trades)


# ==========================================
# 2. 生成模拟数据 (Simulate Volatile Market)
# ==========================================
np.random.seed(42)
days = 250
# 模拟一个震荡向上的高波动 ETF (如半导体)
# 均值漂移 mu=0.05, 波动率 sigma=0.40 (非常高)
returns = np.random.normal(0.05 / 250, 0.40 / np.sqrt(250), days)
price_path = 1.0 * np.exp(np.cumsum(returns))

# 强行制造一个"深V"反转，测试网格抗压能力
# 前100天跌，后150天涨
t = np.linspace(0, 10, days)
trend = np.sin(t) * 0.2
price_path = price_path + trend
price_path = price_path / price_path[0]  # 归一化从 1.0 开始

# ==========================================
# 3. 运行回测
# ==========================================
# 初始资金 10万，网格 2%，每格交易 5000元
strategy = GridStrategy(price_path, initial_capital=100000, grid_size=0.02, trade_amount=5000)
portfolio_curve, trade_log = strategy.run_backtest()

# 基准收益 (Buy & Hold)
benchmark = (price_path / price_path[0]) * 100000

# ==========================================
# 4. 绘图展示
# ==========================================
plt.figure(figsize=(12, 8))

# 子图1: 账户净值对比
plt.subplot(2, 1, 1)
plt.plot(portfolio_curve, label='Grid Strategy (50% Cash Start)', color='red', linewidth=2)
plt.plot(benchmark, label='Buy & Hold (Full Position)', color='gray', linestyle='--', alpha=0.6)
plt.title('Performance: Grid Trading vs Buy & Hold (Simulated Volatile ETF)')
plt.ylabel('Account Value (RMB)')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 交易点位分析
plt.subplot(2, 1, 2)
plt.plot(price_path, label='ETF Price', color='blue', alpha=0.5)
# 标记买卖点
buys = trade_log[trade_log['action'] == 'Buy']
sells = trade_log[trade_log['action'] == 'Sell']
plt.scatter(buys['index'], buys['price'], marker='^', color='green', s=50, label='Buy', zorder=5)
plt.scatter(sells['index'], sells['price'], marker='v', color='red', s=50, label='Sell', zorder=5)
plt.title(f'Trade Execution (Total Trades: {len(trade_log)})')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grid_backtest_result.png')
