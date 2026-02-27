import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. 基础物理引擎 (支持首尾相接与异象注入)
# ==========================================
def generate_gbm_segment(S0, mu, sigma, N, outlier_idx=None, outlier_multiplier=1.0):
    dt = 1.0 / 252  # 年化步长
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)
        if i == outlier_idx:
            S[i] *= outlier_multiplier  # 注入异象
    return S


def generate_ou_segment(S0, theta, mu_level, sigma, N):
    dt = 1.0 / 252
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] + theta * (mu_level - S[i - 1]) * dt + sigma * dW
    return S


def extract_features(path):
    log_rets = np.diff(np.log(path))
    price_dev = (path[:-1] - path[0]) / path[0]
    return np.column_stack([log_rets, price_dev])


# ==========================================
# 2. 预训练三核专家大脑 (在纯净数据上学习物理法则)
# ==========================================
np.random.seed(42)
N_train = 252

feat_bull = np.concatenate([extract_features(generate_gbm_segment(100, 0.5, 0.15, N_train)) for _ in range(30)])
feat_bear = np.concatenate([extract_features(generate_gbm_segment(100, -0.5, 0.15, N_train)) for _ in range(30)])
feat_mr = np.concatenate([extract_features(generate_ou_segment(100, 10.0, 100, 15, N_train)) for _ in range(30)])
lengths = [N_train - 1] * 30

hmm_bull = GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(feat_bull, lengths)
hmm_bear = GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(feat_bear, lengths)
hmm_mr = GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(feat_mr, lengths)

# ==========================================
# 3. 构建 5 段混合测试行情 (温室的边缘)
# ==========================================
N_seg = 100
path_total = []
true_states = []

# Phase 1: 震荡 (OU)
p1 = generate_ou_segment(100, 8.0, 100, 15, N_seg)
path_total.extend(p1);
true_states.extend(['MR'] * N_seg)

# Phase 2: 主升浪 (GBM mu=0.6) + 第50天跌6%
p2 = generate_gbm_segment(p1[-1], 0.6, 0.15, N_seg, outlier_idx=50, outlier_multiplier=0.94)
path_total.extend(p2);
true_states.extend(['BULL'] * N_seg)

# Phase 3: 震荡 (OU，平衡点设在P2终点)
p3 = generate_ou_segment(p2[-1], 8.0, p2[-1], 15, N_seg)
path_total.extend(p3);
true_states.extend(['MR'] * N_seg)

# Phase 4: 主跌浪 (GBM mu=-0.5) + 第50天涨7%逼空
p4 = generate_gbm_segment(p3[-1], -0.5, 0.15, N_seg, outlier_idx=50, outlier_multiplier=1.07)
path_total.extend(p4);
true_states.extend(['BEAR'] * N_seg)

# Phase 5: 慢牛 (GBM mu=0.3)
p5 = generate_gbm_segment(p4[-1], 0.3, 0.15, N_seg)
path_total.extend(p5);
true_states.extend(['BULL'] * N_seg)

path_total = np.array(path_total)

# ==========================================
# 4. 流式前向预测 (Rolling Window Inference)
# ==========================================
window_size = 60  # 60天滚动切片 (必须大于特征维度要求)
predicted_states = ['UNKNOWN'] * window_size  # 初始窗口内无法预测

for t in range(window_size, len(path_total)):
    # 严禁使用 t 之后的数据！只提取过去 window_size 的切片
    window_slice = path_total[t - window_size: t + 1]
    feat = extract_features(window_slice)

    # 三个专家竞争打分
    score_bull = hmm_bull.score(feat)
    score_bear = hmm_bear.score(feat)
    score_mr = hmm_mr.score(feat)

    scores = {'BULL': score_bull, 'BEAR': score_bear, 'MR': score_mr}
    predicted_states.append(max(scores, key=scores.get))

# ==========================================
# 5. 专业可视化面板 (Tear Sheet)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('Walk-Forward HMM Regime Detection (Strict No Look-Ahead Bias)', fontsize=18, fontweight='bold')

# 上图：价格曲线与预测状态背景
ax1.plot(path_total, color='black', linewidth=1.5, label='Price Path')

# 标记异常值点
ax1.scatter([150], [path_total[150]], color='purple', s=100, marker='v', zorder=5, label='-6% Washout (Bull)')
ax1.scatter([350], [path_total[350]], color='orange', s=100, marker='^', zorder=5, label='+7% Short Squeeze (Bear)')

# 用背景色绘制 HMM 的前向预测结果
for t in range(window_size, len(path_total)):
    if predicted_states[t] == 'BULL':
        ax1.axvspan(t - 1, t, color='#ffcccc', alpha=0.6, lw=0)
    elif predicted_states[t] == 'BEAR':
        ax1.axvspan(t - 1, t, color='#ccffcc', alpha=0.6, lw=0)
    elif predicted_states[t] == 'MR':
        ax1.axvspan(t - 1, t, color='#ccccff', alpha=0.6, lw=0)

# 画出真实的阶段分割线
for idx in [100, 200, 300, 400]:
    ax1.axvline(idx, color='grey', linestyle='--', linewidth=1)

ax1.set_title('Price & Predicted Regime (Red=Bull, Green=Bear, Blue=MR)', fontsize=12)
ax1.set_ylabel('Asset Price')
ax1.legend(loc='upper left')
ax1.set_xlim(0, 500)

# 下图：真实状态的对比 (Ground Truth)
state_to_num = {'MR': 0, 'BULL': 1, 'BEAR': -1, 'UNKNOWN': 0}
true_num = [state_to_num[s] for s in true_states]
pred_num = [state_to_num[s] for s in predicted_states]

ax2.plot(true_num, label='Ground Truth (Theoretical)', color='black', linestyle='--', linewidth=2)
ax2.plot(pred_num, label='HMM Walk-Forward Prediction', color='blue', linewidth=1.5)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['BEAR (-1)', 'MR (0)', 'BULL (1)'])
ax2.set_title('Regime State Comparison: Delayed Reaction & Whipsaw Wounds', fontsize=12)
ax2.legend(loc='lower right')
ax2.set_xlim(0, 500)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("../pic/hmm_walk_forward.png", dpi=300)
plt.show()