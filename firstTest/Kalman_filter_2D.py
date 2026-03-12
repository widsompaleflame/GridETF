import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 基础物理引擎 (纯净模拟数据)
# ==========================================
def generate_gbm_segment(S0, mu, sigma, N, outlier_idx=None, outlier_multiplier=1.0):
    dt = 1.0 / 252
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)
        if i == outlier_idx:
            S[i] *= outlier_multiplier
    return S


def generate_ou_segment(S0, theta, mu_level, sigma, N):
    dt = 1.0 / 252
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] + theta * (mu_level - S[i - 1]) * dt + sigma * dW
    return S


# ==========================================
# 2. 构建 5 段混合测试行情
# ==========================================
np.random.seed(42)
N_seg = 100
path_total = []

p1 = generate_ou_segment(100, 8.0, 100, 15, N_seg)  # 震荡
p2 = generate_gbm_segment(p1[-1], 0.6, 0.15, N_seg, 50, 0.94)  # 主升 + 洗盘
p3 = generate_ou_segment(p2[-1], 8.0, p2[-1], 15, N_seg)  # 震荡
p4 = generate_gbm_segment(p3[-1], -0.5, 0.15, N_seg, 50, 1.07)  # 主跌 + 逼空
p5 = generate_gbm_segment(p4[-1], 0.3, 0.15, N_seg)  # 慢牛
path_total = np.concatenate([p1, p2, p3, p4, p5])

# ==========================================
# 3. 二维卡尔曼滤波器核心实现 (严格无未来函数)
# ==========================================
N = len(path_total)
Z = path_total  # 观测值序列

# 状态转移矩阵 F (位置 + 速度)
F = np.array([[1.0, 1.0],
              [0.0, 1.0]])

# 观测矩阵 H (只能观测到位置/价格)
H = np.array([[1.0, 0.0]])

# 系统噪声协方差 Q (极度危险的超参数！)
# Q越大，模型越信任新数据，反应越快但也越毛躁
q_variance = 1e-4
Q = np.array([[q_variance, 0.0],
              [0.0, q_variance]])

# 观测噪声方差 R
# R越大，模型认为市场越疯狂，越倾向于维持自身的平滑预测
R = np.array([[0.05]])

# 初始化状态估计 x 和 误差协方差 P
x_hat = np.array([[Z[0]], [0.0]])  # 初始假设：位置在P0，速度为0
P = np.eye(2) * 1.0

# 记录结果的容器
kf_prices = np.zeros(N)
kf_velocities = np.zeros(N)

for t in range(N):
    # --- 预测阶段 (Predict) ---
    # 利用 t-1 的后验状态预测 t 时刻的先验状态
    x_pred = F @ x_hat
    P_pred = F @ P @ F.T + Q

    # --- 更新阶段 (Update) ---
    # 接收 t 时刻的真实观测值 Z[t]
    z_t = np.array([[Z[t]]])

    # 计算卡尔曼增益 K
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # 吸收测量残差 (Innovation)，修正状态估计
    y = z_t - H @ x_pred
    x_hat = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

    # 保存 t 时刻的后验状态
    kf_prices[t] = x_hat[0, 0]
    kf_velocities[t] = x_hat[1, 0]

# ==========================================
# 4. 交易信号生成逻辑 (基于隐状态 Velocity)
# ==========================================
df = pd.DataFrame({'Price': Z, 'KF_Price': kf_prices, 'Velocity': kf_velocities})

# 设定速度的静区阈值 (Deadband Threshold)，过滤震荡市的微小噪声
threshold = 0.05

df['Signal'] = 0
df.loc[df['Velocity'] > threshold, 'Signal'] = 1  # 做多
df.loc[df['Velocity'] < -threshold, 'Signal'] = -1  # 做空

# 寻找买卖翻转点 (交叉点)
df['Position_Change'] = df['Signal'].diff()
buy_signals = df[df['Position_Change'] == 2]  # 从 -1 变 1，或 0 变 1 (粗略过滤)
sell_signals = df[df['Position_Change'] == -2]  # 从 1 变 -1，或 0 变 -1

# ==========================================
# 5. 专业可视化面板 (Tear Sheet)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2.5, 1]})
fig.suptitle('2D Kinematic Kalman Filter: Momentum Engine & Regime Detection', fontsize=18, fontweight='bold')

# 上图：价格曲线与卡尔曼平滑线
ax1.plot(df.index, df['Price'], color='black', alpha=0.5, linewidth=1, label='Noisy Price (Observation)')
ax1.plot(df.index, df['KF_Price'], color='blue', linewidth=2, label='KF Smoothed True Price')

# 标记买卖点
ax1.scatter(buy_signals.index, df.loc[buy_signals.index, 'KF_Price'], marker='^', color='green', s=120, zorder=5,
            label='Buy Signal (Velocity > Threshold)')
ax1.scatter(sell_signals.index, df.loc[sell_signals.index, 'KF_Price'], marker='v', color='red', s=120, zorder=5,
            label='Sell Signal (Velocity < -Threshold)')

ax1.set_title('Asset Price and Kalman State Estimation', fontsize=12)
ax1.set_ylabel('Asset Price')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlim(0, 500)

# 下图：隐状态 —— 动量引擎 (Velocity)
ax2.plot(df.index, df['Velocity'], color='purple', linewidth=1.5, label='Hidden State: Velocity (Momentum)')
ax2.axhline(threshold, color='green', linestyle='--', alpha=0.5, label='Long Threshold')
ax2.axhline(-threshold, color='red', linestyle='--', alpha=0.5, label='Short Threshold')
ax2.axhline(0, color='black', linewidth=1)

# 用背景色绘制卡尔曼滤波器的趋势判断
for t in range(1, len(df)):
    if df['Signal'].iloc[t] == 1:
        ax2.axvspan(t - 1, t, color='#ccffcc', alpha=0.4, lw=0)
    elif df['Signal'].iloc[t] == -1:
        ax2.axvspan(t - 1, t, color='#ffcccc', alpha=0.4, lw=0)

ax2.set_title('Hidden State Extraction: The Kinematic Velocity Engine', fontsize=12)
ax2.set_ylabel('Velocity (Unit/dt)')
ax2.legend(loc='upper left')
ax2.set_xlim(0, 500)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('../pic/Kalman_filter_2D.png', dpi=300)
plt.show()