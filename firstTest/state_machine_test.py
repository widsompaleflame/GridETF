"""
两分状态机:震荡/趋势的区分:
赫斯特指数在这一百张图片里的结果,51个判断正确,49个判断错误,那么这个赫斯特指数跟扔硬币的概率没有任何区别,无法分辨趋势和震荡,ToT,
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 蒙特卡洛路径生成器 (SDE Models)
# ==========================================
def generate_gbm_path(S0, mu, sigma, T, N):
    """几何布朗运动 (GBM) - 模拟趋势"""
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return S


def generate_ou_path(S0, theta, mu, sigma, T, N):
    """Ornstein-Uhlenbeck 过程 (OU) - 模拟均值回归(震荡)"""
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        # dx_t = theta * (mu - x_t)dt + sigma * dW_t
        S[i] = S[i - 1] + theta * (mu - S[i - 1]) * dt + sigma * dW
    return S


# ==========================================
# 2. 状态机核心：赫斯特指数 (Hurst Exponent) 计算器
# ==========================================
def calculate_hurst(ts):
    """
    使用简化的重标极差分析 (R/S Analysis) 的变体来估算 Hurst 指数
    """
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # 对数线性回归: log(tau) = H * log(lag) + C
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # 乘以 2 因为我们用了标准差而不是方差
    H = poly[0] * 2.0
    return H


# ==========================================
# 3. 运行 100 次蒙特卡洛并进行状态机分类
# ==========================================
np.random.seed(42)
num_paths = 100
N_steps = 252  # 一年252个交易日

paths = []
true_labels = []
predicted_h = []

for i in range(num_paths):
    # 50% 概率生成趋势 (GBM)，50% 概率生成震荡 (OU)
    if np.random.rand() > 0.5:
        # 趋势：高漂移，低波动
        path = generate_gbm_path(S0=100, mu=0.2, sigma=0.15, T=1, N=N_steps)
        true_labels.append("Trend_GBM")
    else:
        # 震荡：强均值回归引力 (theta大)
        path = generate_ou_path(S0=100, theta=10.0, mu=100, sigma=15, T=1, N=N_steps)
        true_labels.append("Mean-Rev_OU")

    paths.append(path)

    # 状态机进行计算
    H = calculate_hurst(path)
    predicted_h.append(H)

# ==========================================
# 4. 可视化：10x10 图表矩阵
# ==========================================
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
fig.suptitle('Monte Carlo Regime Switching State Machine (100 Paths Classification)', fontsize=24, fontweight='bold')
result_list=[]

for i, ax in enumerate(axes.flatten()):
    ax.plot(paths[i], color='black', linewidth=0.8)

    H_val = predicted_h[i]

    # 状态机判定逻辑
    if H_val > 0.55:
        state = "Trend_GBM"
        bg_color = '#ffe6e6'  # 浅红色代表趋势
        if true_labels[i] == "Trend_GBM":
            result_list.append("right")
        else:
            result_list.append("wrong")
    elif H_val < 0.45:
        state = "Mean-Rev_OU"
        bg_color = '#e6f2ff'  # 浅蓝色代表震荡
        if true_labels[i] == "Mean-Rev_OU":
            result_list.append("right")
        else:
            result_list.append("wrong")
    else:
        state = "RANDOM"
        bg_color = '#f2f2f2'  # 灰色代表随机游走(无法判断)
        result_list.append("wrong")

    ax.set_facecolor(bg_color)

    # 去除坐标轴以保持画面整洁
    ax.set_xticks([])
    ax.set_yticks([])

    # 标注 Hurst 指数和分类结果
    ax.set_title(f'H: {H_val:.2f}\n{state}\nreal_label:{true_labels[i]}', fontsize=8, pad=2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../pic/100_monte_carlo_regimes.png', dpi=150)
plt.show()

print(f"Right的数量:{result_list.count('right')}")
print(f"Wrong的数量:{result_list.count('wrong')}")