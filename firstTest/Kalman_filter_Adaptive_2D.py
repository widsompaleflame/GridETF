import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 基础物理引擎 (每次运行生成全新随机路径)
# ==========================================
def generate_gbm_segment(S0, mu, sigma, N, outlier_idx=None, outlier_multiplier=1.0):
    dt = 1.0 / 252
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)
        if outlier_idx is not None and i == outlier_idx:
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
# 2. 构建动态混合测试行情 (不设固定Seed)
# ==========================================
N_seg = 100
path_total = []

# Phase 1: 震荡
p1 = generate_ou_segment(100, 8.0, 100, 15, N_seg)
# Phase 2: 主升浪 + 随机位置洗盘
p2 = generate_gbm_segment(p1[-1], 0.6, 0.15, N_seg, np.random.randint(10, 90), 0.94)
# Phase 3: 震荡
p3 = generate_ou_segment(p2[-1], 8.0, p2[-1], 15, N_seg)
# Phase 4: 主跌浪 + 随机位置逼空
p4 = generate_gbm_segment(p3[-1], -0.5, 0.15, N_seg, np.random.randint(10, 90), 1.07)
# Phase 5: 慢牛
p5 = generate_gbm_segment(p4[-1], 0.3, 0.15, N_seg)

path_total = np.concatenate([p1, p2, p3, p4, p5])

# ==========================================
# 3. 自适应二维卡尔曼滤波器 (AKF - Adaptive Q/R)
# ==========================================
N = len(path_total)
Z = path_total

# 状态转移矩阵 F (位置 + 速度)
F = np.array([[1.0, 1.0],
              [0.0, 1.0]])

# 观测矩阵 H (只能观测到位置)
H = np.array([[1.0, 0.0]])

# 初始化超参数与状态
# 此时的 Q 和 R 是基准值，算法会在迭代中自适应修改它们
base_q = 1e-5
Q = np.array([[base_q, 0.0],
              [0.0, base_q]])
base_r = 0.01
R = np.array([[base_r]])

x_hat = np.array([[Z[0]], [0.0]])
P = np.eye(2) * 1.0

kf_prices = np.zeros(N)
kf_velocities = np.zeros(N)

# 自适应窗口变量
innovations = []
adaptive_window = 10  # 用于计算近期误差方差的滚动窗口

for t in range(N):
    # --- 1. 先验预测 (Predict) ---
    x_pred = F @ x_hat
    P_pred = F @ P @ F.T + Q

    # 接收最新报价
    z_t = np.array([[Z[t]]])

    # --- 2. 计算残差 (Innovation) ---
    y = z_t - H @ x_pred  # 实际观测值 与 模型预测值 的偏差
    innovations.append(y[0, 0])

    # --- 3. 动态调整 Q 和 R (自适应核心) ---
    # 如果近期残差的方差激增，说明模型跟不上市场变化 (可能是趋势突变或波动率放大)
    if len(innovations) >= adaptive_window:
        recent_err_var = np.var(innovations[-adaptive_window:])
        # 启发式规则：当误差显著大于基准观测方差时，增大 Q (相信新数据)，同时微调 R
        if recent_err_var > base_r * 2:
            Q = np.eye(2) * (base_q * (recent_err_var / base_r))
            R = np.array([[recent_err_var * 0.5]])
        else:
            Q = np.eye(2) * base_q
            R = np.array([[base_r]])

    # --- 4. 后验更新 (Update) ---
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_hat = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

    kf_prices[t] = x_hat[0, 0]
    kf_velocities[t] = x_hat[1, 0]

# ==========================================
# 4. 基于自适应动量的 Z-Score 信号生成
# ==========================================
df = pd.DataFrame({'Price': Z, 'KF_Price': kf_prices, 'Velocity': kf_velocities})

# 风险修正：使用动态滚动标准差将绝对 Velocity 转换为 Z-Score
# 这解决了固定阈值在不同波动率下失效的问题
rolling_vol = df['Velocity'].rolling(window=20, min_periods=5).std()
df['Vel_Z_Score'] = df['Velocity'] / (rolling_vol + 1e-6)  # 加小数防除零

# 设定 Z-Score 阈值 (无量纲)
z_threshold = 1.2

df['Signal'] = 0
df.loc[df['Vel_Z_Score'] > z_threshold, 'Signal'] = 1
df.loc[df['Vel_Z_Score'] < -z_threshold, 'Signal'] = -1

df['Position_Change'] = df['Signal'].diff()
buy_signals = df[df['Position_Change'] >= 1]
sell_signals = df[df['Position_Change'] <= -1]

# ==========================================
# 5. 专业可视化面板
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2.5, 1]})
fig.suptitle('Adaptive 2D Kalman Filter: Dynamic Q/R & Z-Score Momentum', fontsize=16, fontweight='bold')

# 上图：价格曲线与自适应卡尔曼平滑线
ax1.plot(df.index, df['Price'], color='black', alpha=0.4, linewidth=1, label='Noisy Price (GBM+OU)')
ax1.plot(df.index, df['KF_Price'], color='blue', linewidth=2, label='AKF Smoothed Price')

# 标记买卖点
ax1.scatter(buy_signals.index, df.loc[buy_signals.index, 'KF_Price'], marker='^', color='green', s=80, zorder=5,
            label='Buy (Z-Score > Threshold)')
ax1.scatter(sell_signals.index, df.loc[sell_signals.index, 'KF_Price'], marker='v', color='red', s=80, zorder=5,
            label='Sell (Z-Score < -Threshold)')

ax1.set_title('Asset Price and Adaptive Kalman State Estimation', fontsize=12)
ax1.set_ylabel('Asset Price')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# 下图：隐状态 —— 自适应 Z-Score 动量
ax2.plot(df.index, df['Vel_Z_Score'], color='purple', linewidth=1.5, label='Hidden State: Velocity Z-Score')
ax2.axhline(z_threshold, color='green', linestyle='--', alpha=0.5)
ax2.axhline(-z_threshold, color='red', linestyle='--', alpha=0.5)
ax2.axhline(0, color='black', linewidth=1)

for t in range(1, len(df)):
    if df['Signal'].iloc[t] == 1:
        ax2.axvspan(t - 1, t, color='#ccffcc', alpha=0.3, lw=0)
    elif df['Signal'].iloc[t] == -1:
        ax2.axvspan(t - 1, t, color='#ffcccc', alpha=0.3, lw=0)

ax2.set_title('Normalized Momentum: Volatility-Adjusted Z-Score', fontsize=12)
ax2.set_ylabel('Z-Score')
ax2.legend(loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('../pic/Kalman_filter_Adaptive_2d.png',dpi=300)
plt.show()