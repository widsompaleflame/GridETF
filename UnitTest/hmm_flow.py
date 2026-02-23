"""
这个脚本将只生成 1条熊市趋势 (Bear Trend) 和 1条震荡 (Mean-Reverting)。它会将整个生命周期（原始路径 $\rightarrow$ 提取特征 $\rightarrow$ HMM 内部参数 $\rightarrow$ 动态评分概率）全部画出来。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 1. 核心生成器
def generate_gbm_path(S0, mu, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

def generate_ou_path(S0, theta, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i-1] + theta * (mu - S[i-1]) * dt + sigma * dW
    return S

def extract_features(path):
    log_returns = np.diff(np.log(path))
    price_deviation = (path[:-1] - path[0]) / path[0]
    return np.column_stack([log_returns, price_deviation])

# 设置随机种子以复现这幅特定的教学图
np.random.seed(10)
N_steps = 252

# 2. 生成两条用于透视的路径 (一条彻头彻尾的熊市，一条震荡)
# 熊市: mu = -0.5
path_bear = generate_gbm_path(100, -0.5, 0.15, 1, N_steps)
# 震荡: theta = 10 (回归力度强)
path_ou = generate_ou_path(100, 10.0, 100, 15, 1, N_steps)

feat_bear = extract_features(path_bear)
feat_ou = extract_features(path_ou)

# 3. 训练两个 HMM 模型 (此处为了展示，仅使用单条路径训练，实战需多条)
hmm_trend = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
hmm_trend.fit(feat_bear, [N_steps-1])

hmm_mr = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
hmm_mr.fit(feat_ou, [N_steps-1])

# 计算测试得分
score_bear_on_trend = hmm_trend.score(feat_bear)
score_bear_on_mr = hmm_mr.score(feat_bear)
score_ou_on_trend = hmm_trend.score(feat_ou)
score_ou_on_mr = hmm_mr.score(feat_ou)

# 4. 绘图：四维可视化
fig = plt.figure(figsize=(18, 14))
fig.suptitle('HMM X-Ray: From Raw Data to Probability Matrices', fontsize=20, fontweight='bold')

# --- Row 1: 原始价格序列 ---
ax1 = plt.subplot(3, 2, 1)
ax1.plot(path_bear, color='red', label=f'GBM Bear Market ($\mu=-0.5$)')
ax1.set_title('Raw Price: Downward Trend', fontsize=14)
ax1.legend()

ax2 = plt.subplot(3, 2, 2)
ax2.plot(path_ou, color='blue', label=f'OU Mean-Reverting ($\mu=100$)')
ax2.set_title('Raw Price: Oscillation', fontsize=14)
ax2.legend()

# --- Row 2: 提取出的特征相空间 (Feature Phase Space) ---
# 这是 HMM 真正"看"到的世界
ax3 = plt.subplot(3, 2, 3)
ax3.scatter(feat_bear[:, 1], feat_bear[:, 0], alpha=0.5, color='red', s=15)
ax3.set_title('Features: Log-Return vs Price Deviation', fontsize=12)
ax3.set_xlabel('Price Deviation (x-axis)')
ax3.set_ylabel('Log Returns (y-axis)')
ax3.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax3.axvline(0, color='black', linestyle='--', linewidth=0.5)

ax4 = plt.subplot(3, 2, 4)
ax4.scatter(feat_ou[:, 1], feat_ou[:, 0], alpha=0.5, color='blue', s=15)
ax4.set_title('Features: Notice the negative correlation tilt!', fontsize=12)
ax4.set_xlabel('Price Deviation (x-axis)')
ax4.set_ylabel('Log Returns (y-axis)')
ax4.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax4.axvline(0, color='black', linestyle='--', linewidth=0.5)

# --- Row 3: HMM 提取的转移概率矩阵 (Transition Matrix) ---
ax5 = plt.subplot(3, 2, 5)
sns.heatmap(hmm_trend.transmat_, annot=True, cmap='Reds', cbar=False, ax=ax5, fmt='.2f')
ax5.set_title(f'Trend Model Transition Matrix $\mathbf{{A}}$\nScore on itself: {score_bear_on_trend:.1f} | Score on MR: {score_bear_on_mr:.1f}', fontsize=12)
ax5.set_xlabel('To State')
ax5.set_ylabel('From State')

ax6 = plt.subplot(3, 2, 6)
sns.heatmap(hmm_mr.transmat_, annot=True, cmap='Blues', cbar=False, ax=ax6, fmt='.2f')
ax6.set_title(f'MR Model Transition Matrix $\mathbf{{A}}$\nScore on itself: {score_ou_on_mr:.1f} | Score on Trend: {score_ou_on_trend:.1f}', fontsize=12)
ax6.set_xlabel('To State')
ax6.set_ylabel('From State')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../pic/hmm_flow.png', dpi=150)
plt.show()

# 打印 HMM 学习到的高斯分布均值 (Emission Means)
print("\n=== HMM 学习到的高斯分布参数 ===")
print("【趋势模型 (熊市)】学习到的观测特征均值 [Log Return, Price Deviation]:")
for i, mean in enumerate(hmm_trend.means_):
    print(f"隐状态 {i}: {mean}")

print("\n【震荡模型】学习到的观测特征均值 [Log Return, Price Deviation]:")
for i, mean in enumerate(hmm_mr.means_):
    print(f"隐状态 {i}: {mean}")