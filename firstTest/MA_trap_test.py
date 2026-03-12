import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础物理引擎 (完全保留你的原始设定)
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
# 2. 构建 5 段混合测试行情 (温室的边缘)
# ==========================================
np.random.seed(42)
N_seg = 100
path_total = []
true_states = []

# Phase 1: 震荡 (OU)
p1 = generate_ou_segment(100, 8.0, 100, 15, N_seg)
path_total.extend(p1)
true_states.extend(['MR'] * N_seg)

# Phase 2: 主升浪 (GBM mu=0.6) + 第50天跌6%
p2 = generate_gbm_segment(p1[-1], 0.6, 0.15, N_seg, outlier_idx=50, outlier_multiplier=0.94)
path_total.extend(p2)
true_states.extend(['BULL'] * N_seg)

# Phase 3: 震荡 (OU，平衡点设在P2终点)
p3 = generate_ou_segment(p2[-1], 8.0, p2[-1], 15, N_seg)
path_total.extend(p3)
true_states.extend(['MR'] * N_seg)

# Phase 4: 主跌浪 (GBM mu=-0.5) + 第50天涨7%逼空
p4 = generate_gbm_segment(p3[-1], -0.5, 0.15, N_seg, outlier_idx=50, outlier_multiplier=1.07)
path_total.extend(p4)
true_states.extend(['BEAR'] * N_seg)

# Phase 5: 慢牛 (GBM mu=0.3)
p5 = generate_gbm_segment(p4[-1], 0.3, 0.15, N_seg)
path_total.extend(p5)
true_states.extend(['BULL'] * N_seg)

# ==========================================
# 3. 均线状态机与无未来函数信号生成
# ==========================================
df = pd.DataFrame({'price': path_total, 'true_state': true_states})

# 设定均线窗口 (典型的过拟合重灾区，这里取10和30)
fast_w = 10
slow_w = 30

# 严格无未来计算：t时刻的均值只包含 t-(w-1) 到 t 的数据
df['MA_Fast'] = df['price'].rolling(window=fast_w, min_periods=fast_w).mean()
df['MA_Slow'] = df['price'].rolling(window=slow_w, min_periods=slow_w).mean()

# 状态判定：快线 > 慢线 视为 BULL (1)，否则视为 BEAR (-1)
# 注意：均线系统无法像HMM一样输出 MR (震荡) 状态，这是其先天缺陷
df['predicted_state'] = np.where(df['MA_Fast'] >= df['MA_Slow'], 1, -1)
# 填充初始缺失期的状态为 0 (UNKNOWN)
df.loc[df['MA_Slow'].isna(), 'predicted_state'] = 0

# ==========================================
# 4. 专业可视化面板 (Tear Sheet)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f'Walk-Forward Dual MA Regime Detection (Fast={fast_w}, Slow={slow_w})', fontsize=18, fontweight='bold')

# 上图：价格曲线、均线与预测状态背景
ax1.plot(df.index, df['price'], color='black', linewidth=1.5, label='Price Path')
ax1.plot(df.index, df['MA_Fast'], color='blue', linestyle='-.', linewidth=1, label=f'{fast_w}-Day MA')
ax1.plot(df.index, df['MA_Slow'], color='red', linestyle='-.', linewidth=1, label=f'{slow_w}-Day MA')

# 标记异常值点
ax1.scatter([150], [df['price'].iloc[150]], color='purple', s=100, marker='v', zorder=5, label='-6% Washout (Bull)')
ax1.scatter([350], [df['price'].iloc[350]], color='orange', s=100, marker='^', zorder=5, label='+7% Short Squeeze (Bear)')

# 用背景色绘制均线的前向预测结果
for t in range(slow_w, len(df)):
    if df['predicted_state'].iloc[t] == 1:
        ax1.axvspan(t - 1, t, color='#ffcccc', alpha=0.6, lw=0)  # BULL
    elif df['predicted_state'].iloc[t] == -1:
        ax1.axvspan(t - 1, t, color='#ccffcc', alpha=0.6, lw=0)  # BEAR

# 画出真实的阶段分割线
for idx in [100, 200, 300, 400]:
    ax1.axvline(idx, color='grey', linestyle='--', linewidth=1)

ax1.set_title('Price, Moving Averages & Predicted Regime (Red=Bull, Green=Bear)', fontsize=12)
ax1.set_ylabel('Asset Price')
ax1.legend(loc='upper left')
ax1.set_xlim(0, 500)

# 下图：真实状态的对比 (Ground Truth vs MA)
state_to_num = {'MR': 0, 'BULL': 1, 'BEAR': -1}
df['true_num'] = df['true_state'].map(state_to_num)

ax2.plot(df.index, df['true_num'], label='Ground Truth (Theoretical)', color='black', linestyle='--', linewidth=2)
ax2.plot(df.index, df['predicted_state'], label='Dual MA Prediction', color='blue', linewidth=1.5)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['BEAR (-1)', 'MR (0)', 'BULL (1)'])
ax2.set_title('Regime State Comparison: The "Whipsaw" Reality in MR Zones', fontsize=12)
ax2.legend(loc='lower right')
ax2.set_xlim(0, 500)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('../pic/MA_trap.png',dpi=300)
plt.show()