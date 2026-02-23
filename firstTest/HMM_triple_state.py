"""
非对称波动率陷阱（Asymmetric Volatility）与杠杆效应（Leverage Effect）。
您提到的“牛市多暴跌”，在金融工程中被称为左偏肥尾（Left-Skewed Fat Tail）。
在标准的高斯隐马尔可夫模型（Gaussian HMM）中，
观测值的发射概率服从正态分布：$$P(O_t | S_j) = \frac{1}{\sqrt{2\pi|\Sigma_j|}} \exp\left(-\frac{1}{2}(O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j)\right)$$

注意公式中的指数项。如果出现一个单日 -5% 的洗盘（对于日均波动率 1% 的市场来说，这是 5 倍标准差的 $5\sigma$ 事件），
指数项会被放大 $5^2 = 25$ 倍。这会导致该条路径在“牛市模型”下的似然概率瞬间坍塌趋近于 0（即 $\log P \to -\infty$）。
机器会因为极度“畏惧”这一天的离群值，而否定剩下 251 天的上涨趋势，从而产生您所说的 Whipsaw（锯齿放血）效应。
为了用数据验证您的直觉，我为您编写了这套**“三模型极大似然竞争架构”**的压力测试脚本。我们将生成 150 条路径，
并在其中 10 条趋势路径中人工注入 5% 的单日反向洗盘黑天鹅。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. 蒙特卡洛 SDE 生成器 (含黑天鹅注入机制)
# ==========================================
def generate_gbm_path(S0, mu, sigma, T, N, outlier=None):
    """
    outlier: 'bull_drop' (牛市-5%洗盘) 或 'bear_spike' (熊市+5%逼空)
    """
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

        # 在第 126 天 (约半年处) 强行注入单日极端波动
        if i == N // 2:
            if outlier == 'bull_drop':
                S[i] *= 0.95  # 单日暴跌 5%
            elif outlier == 'bear_spike':
                S[i] *= 1.05  # 单日暴涨 5%
    return S


def generate_ou_path(S0, theta, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] + theta * (mu - S[i - 1]) * dt + sigma * dW
    return S


def extract_features(path):
    log_returns = np.diff(np.log(path))
    price_deviation = (path[:-1] - path[0]) / path[0]
    return np.column_stack([log_returns, price_deviation])


# ==========================================
# 2. 训练三个独立的 HMM 专家模型 (纯净数据)
# ==========================================
np.random.seed(42)
N_steps = 252
num_train = 50

# 生成纯净的训练集
train_bull = [generate_gbm_path(100, 0.5, 0.15, 1, N_steps) for _ in range(num_train)]
train_bear = [generate_gbm_path(100, -0.5, 0.15, 1, N_steps) for _ in range(num_train)]
train_mr = [generate_ou_path(100, 10.0, 100, 15, 1, N_steps) for _ in range(num_train)]

# 特征提取与拼接
feat_bull = np.concatenate([extract_features(p) for p in train_bull])
feat_bear = np.concatenate([extract_features(p) for p in train_bear])
feat_mr = np.concatenate([extract_features(p) for p in train_mr])
lengths = [N_steps - 1] * num_train

# 实例化并训练三核引擎
print("正在训练多头、空头与震荡模型...")
hmm_bull = GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(feat_bull, lengths)
hmm_bear = GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(feat_bear, lengths)
hmm_mr = GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(feat_mr, lengths)

# ==========================================
# 3. 构造 150 种实战测试场景 (含 10 个陷阱)
# ==========================================
test_paths = []
test_labels = []  # 真实的底层物理状态
test_outliers = []  # 是否包含黑天鹅

# 45个纯净牛市 + 5个带暴跌的牛市
for _ in range(45):
    test_paths.append(generate_gbm_path(100, 0.5, 0.15, 1, N_steps))
    test_labels.append("BULL");
    test_outliers.append(False)
for _ in range(5):
    test_paths.append(generate_gbm_path(100, 0.5, 0.15, 1, N_steps, outlier='bull_drop'))
    test_labels.append("BULL");
    test_outliers.append(True)

# 45个纯净熊市 + 5个带逼空的熊市
for _ in range(45):
    test_paths.append(generate_gbm_path(100, -0.5, 0.15, 1, N_steps))
    test_labels.append("BEAR");
    test_outliers.append(False)
for _ in range(5):
    test_paths.append(generate_gbm_path(100, -0.5, 0.15, 1, N_steps, outlier='bear_spike'))
    test_labels.append("BEAR");
    test_outliers.append(True)

# 50个纯净震荡
for _ in range(50):
    test_paths.append(generate_ou_path(100, 10.0, 100, 15, 1, N_steps))
    test_labels.append("MR");
    test_outliers.append(False)

# ==========================================
# 4. 竞争打分与性能统计
# ==========================================
results = []
outlier_cases_to_plot = []

for i, path in enumerate(test_paths):
    feat = extract_features(path)

    # 极值似然竞争
    score_bull = hmm_bull.score(feat)
    score_bear = hmm_bear.score(feat)
    score_mr = hmm_mr.score(feat)

    scores = {"BULL": score_bull, "BEAR": score_bear, "MR": score_mr}
    predicted = max(scores, key=scores.get)

    true_label = test_labels[i]
    is_outlier = test_outliers[i]
    is_correct = (predicted == true_label)

    results.append({'True': true_label, 'Pred': predicted, 'Is_Outlier': is_outlier, 'Correct': is_correct})

    # 收集带有黑天鹅陷阱的路径用于绘图
    if is_outlier:
        outlier_cases_to_plot.append((path, true_label, predicted, scores))

df_res = pd.DataFrame(results)

# 统计分析
print("\n=== HMM 压力测试性能报告 ===")
clean_accuracy = df_res[~df_res['Is_Outlier']]['Correct'].mean()
outlier_accuracy = df_res[df_res['Is_Outlier']]['Correct'].mean()

print(f"纯净行情识别准确率 (140条): {clean_accuracy:.1%}")
print(f"黑天鹅行情识别准确率 (10条):  {outlier_accuracy:.1%}")

# ==========================================
# 5. 可视化：透视 10 个被注入洗盘/逼空的路径
# ==========================================
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(f'Whipsaw Effect Inspection: 10 Paths with 5% Intraday Outliers', fontsize=18, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    if i < len(outlier_cases_to_plot):
        path, true_l, pred_l, scores = outlier_cases_to_plot[i]

        ax.plot(path, color='black', linewidth=1.5)
        # 标记黑天鹅发生的位置 (第 126 天)
        ax.axvline(x=126, color='red', linestyle='--', alpha=0.5)

        # 判断颜色：正确为绿，错误为红
        title_color = 'green' if true_l == pred_l else 'red'

        ax.set_title(
            f"True: {true_l} | Pred: {pred_l}\nBull:{scores['BULL']:.0f} Bear:{scores['BEAR']:.0f} MR:{scores['MR']:.0f}",
            color=title_color, fontsize=10)
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"../pic/hmm_triple_state.png", dpi=150)
plt.show()