import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings("ignore")

# 取消固定种子，每次运行生成不同路径
# np.random.seed(42)

N_steps = 252


# ==========================================
# 1. 蒙特卡洛 SDE 生成器 (保持不变)
# ==========================================
def generate_gbm_path(S0, mu, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)


def generate_ou_path(S0, theta, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] + theta * (mu - S[i - 1]) * dt + sigma * dW
    return S


# ==========================================
# 2. 特征工程 (修复版)
# ==========================================
def extract_features(path):
    """
    绝对不能在单条路径内部做 Z-Score！必须保留原始的漂移项和绝对价格水平。
    特征1: 对数收益率 (捕捉趋势漂移)
    特征2: 绝对价格与初始价格的偏离度 (捕捉均值回归引力)
    """
    log_returns = np.diff(np.log(path))
    # 将价格偏离度平移一位对齐
    price_deviation = (path[:-1] - path[0]) / path[0]
    return np.column_stack([log_returns, price_deviation])


# ==========================================
# 3. 训练 HMM 基准模型 (带序列长度标记)
# ==========================================
num_train_paths = 30

# 生成训练数据 (列表格式，防止拼接断层)
train_gbm_paths=[]
for _ in range(num_train_paths):
    # drift=0.5 if np.random.rand()>=0.5 else -0.5 # [GBM] 50% 概率生成向上趋势，50% 概率生成向下趋势
    drift=-0.5
    train_gbm_paths.append(generate_gbm_path(100, drift, 0.15, 1, N_steps))

train_ou_paths = [generate_ou_path(100, 10.0, 100, 15, 1, N_steps) for _ in range(num_train_paths)]

# 提取特征并记录每条序列的长度
feat_gbm = np.concatenate([extract_features(p) for p in train_gbm_paths])
lengths_gbm = [N_steps - 1] * num_train_paths

feat_ou = np.concatenate([extract_features(p) for p in train_ou_paths])
lengths_ou = [N_steps - 1] * num_train_paths

# 训练模型 (传入 lengths 防止接缝处被当成暴跌)
hmm_trend = GaussianHMM(n_components=2, covariance_type="full", n_iter=200)
hmm_trend.fit(feat_gbm, lengths_gbm)

hmm_mr = GaussianHMM(n_components=2, covariance_type="full", n_iter=200)
hmm_mr.fit(feat_ou, lengths_ou)

# ==========================================
# 4. 生成 100 条全新测试路径并分类
# ==========================================
num_paths = 100
paths, true_labels, predicted_labels = [], [], []
correct_count = 0

for i in range(num_paths):
    # 随机选择生成趋势或震荡
    is_trend = np.random.rand() > 0.5
    if is_trend:
        # 这里的参数需要与训练集分布接近
        path = generate_gbm_path(S0=100, mu=0.5, sigma=0.15, T=1, N=N_steps)
        true_label = "TREND"
    else:
        path = generate_ou_path(S0=100, theta=10.0, mu=100, sigma=15, T=1, N=N_steps)
        true_label = "MR"

    paths.append(path)
    true_labels.append(true_label)

    # 提取测试特征
    features = extract_features(path)

    # 使用训练好的两个模型打分 (对数似然)
    score_trend = hmm_trend.score(features)
    score_mr = hmm_mr.score(features)

    pred_label = "TREND" if score_trend > score_mr else "MR"
    predicted_labels.append((pred_label, score_trend, score_mr))

    if pred_label == true_label:
        correct_count += 1

# ==========================================
# 5. 可视化 (与原代码相同，此处略简写核心逻辑)
# ==========================================
fig, axes = plt.subplots(10, 10, figsize=(22, 22))
fig.suptitle(f'HMM Regime Classifier (Accuracy: {correct_count}/100)', fontsize=26, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    ax.plot(paths[i], color='black', linewidth=1)

    true_l = true_labels[i]
    pred_l, s_t, s_m = predicted_labels[i]

    bg_color = '#ffe6e6' if pred_l == "TREND" else '#e6f2ff'

    if true_l != pred_l:
        for spine in ax.spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)
        title_color = 'red'
    else:
        title_color = 'black'

    ax.set_facecolor(bg_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'T: {true_l}\nP: {pred_l}', fontsize=9, color=title_color, pad=2)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('../pic/100_hmm_classifier.png', dpi=150)
plt.show()

print(f"修正后的 HMM 模型分类准确率: {correct_count}%")