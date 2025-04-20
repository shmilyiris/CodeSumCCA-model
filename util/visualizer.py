import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.decomposition import PCA


class Visualizer:
    def __init__(self, csv_path="../result/train.csv"):
        self.csv_path = csv_path
        sns.set(style="whitegrid")

    def visualize_loss(self, data_path=None, save_path=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV 文件不存在: {data_path}")

        # 加载数据
        df = pd.read_csv(data_path)

        if 'train_loss' not in df.columns or 'val_loss' not in df.columns:
            raise ValueError("CSV 中必须包含 'train_loss' 和 'val_loss' 列")

        # 开始绘图
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2, color='#1f77b4')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')

        # 图像美化
        plt.title("Training and Validation Loss", fontsize=16, fontweight='bold')
        plt.xlabel("Epoch", fontsize=13)
        plt.ylabel("Loss", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(df['epoch'])
        plt.legend()
        plt.tight_layout()

        # 保存或展示
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Loss curve saved to {save_path + '_loss.png'}")
        else:
            plt.show()

    def visualize_readability_metrics(self, readability_results: dict):
        """
        readability_results: dict with keys:
        - avg_sentence_length
        - avg_dependency_depth
        - complete_sentence_ratio
        """
        metrics = list(readability_results.keys())
        values = list(readability_results.values())

        plt.figure(figsize=(8, 5))
        sns.barplot(x=metrics, y=values, palette="pastel")
        plt.title("RQ4.1 - Readability Metrics")
        plt.ylabel("Score")
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig("../result/figures/readability_metrics.png")
        plt.show()

    def visualize_accuracy_metrics(self, metrics: dict):
        """
        metrics: dict with keys: 'BLEU', 'METEOR', 'ROUGE-L'
        """
        plt.figure(figsize=(7, 5))
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="Set2")
        plt.title("RQ4.2 - Accuracy Metrics")
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig("../result/figures/accuracy_metrics.png")
        plt.show()

    def visualize_correlation_heatmap(self, project_metrics_df: pd.DataFrame):
        """
        project_metrics_df: DataFrame including BLEU and project features
        """
        corr_matrix = project_metrics_df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("RQ4.3 - Project Features vs. Summary Quality")
        plt.tight_layout()
        plt.savefig("../result/figures/correlation_heatmap.png")
        plt.show()

    def visualize_readability_projects(self):
        # Data
        projects = ['Spring-Framework', 'Dubbo', 'Netty', 'ElasticSearch', 'Maven']
        safm_scores = [0.851, 0.826, 0.832, 0.838, 0.807]
        codebert_scores = [0.824, 0.808, 0.819, 0.813, 0.803]
        hsam_scores = [0.798, 0.776, 0.787, 0.779, 0.790]

        # Create DataFrame for seaborn plotting
        data = {
            'Project': ['Spring-Framework', 'Dubbo', 'Netty', 'ElasticSearch', 'Maven'] * 3,
            'Readability Score': safm_scores + codebert_scores + hsam_scores,
            'Model': ['SAFM'] * 5 + ['CodeBERT'] * 5 + ['HSAM'] * 5
        }

        df = pd.DataFrame(data)

        # Set plot style
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        # Create bar plot
        barplot = sns.barplot(x='Project', y='Readability Score', hue='Model', data=df, palette="muted")

        # Set title and labels in English
        barplot.set_title('Readability Score Distribution Across Five Projects', fontsize=14)
        barplot.set_xlabel('Project', fontsize=12)
        barplot.set_ylabel('Readability Score', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.75, 0.88)

        # Save the figure
        plt.tight_layout()
        plt.savefig('../result/figures/readability_metrics_projects.png')

        # Show the plot
        plt.show()

    def draw_clustering(self, clustering_dict_data, max_points=200):
        """
        可视化聚类结果，将高维数据降维至二维并用颜色区分cluster
        :param clustering_dict_data: dict[int, List[List[float]]] - cluster_id -> list of vectors
        :param max_points: 最大绘制的点数量
        """
        all_points = []
        all_labels = []

        # 收集所有点与标签
        for cluster_id, vectors in clustering_dict_data.items():
            for vec in vectors:
                all_points.append(vec)
                all_labels.append(cluster_id)

        # 数据量太大时进行随机采样
        if len(all_points) > max_points:
            sampled_indices = random.sample(range(len(all_points)), max_points)
            all_points = [all_points[i] for i in sampled_indices]
            all_labels = [all_labels[i] for i in sampled_indices]

        all_points = np.array(all_points)
        all_labels = np.array(all_labels)

        # 使用PCA降维
        pca = PCA(n_components=2)
        reduced_points = pca.fit_transform(all_points)

        # 生成颜色调色板
        unique_clusters = np.unique(all_labels)
        palette = sns.color_palette("hls", len(unique_clusters))
        color_map = {cid: palette[i] for i, cid in enumerate(unique_clusters)}

        # 绘图
        plt.figure(figsize=(8, 6))
        for cid in unique_clusters:
            idx = all_labels == cid
            plt.scatter(
                reduced_points[idx, 0],
                reduced_points[idx, 1],
                s=40,
                alpha=0.7,
                label=f'Cluster {cid}',
                color=color_map[cid]
            )

        plt.title("聚类结果二维可视化（降维+抽样）", fontsize=14)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title="类别", loc="best", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        sns.despine()
        plt.tight_layout()
        plt.show()

    def draw_clustering2(self, clustering_dict_data, max_points=200):
        """
        可视化聚类结果，将高维数据降维至二维并用颜色区分cluster
        :param clustering_dict_data: dict[str, List[List[float]]] - cluster_label -> list of vectors
        :param max_points: 最大绘制的点数量
        """
        all_points = []
        all_labels = []

        # 收集所有点与标签
        for label, vectors in clustering_dict_data.items():
            for vec in vectors:
                all_points.append(vec)
                all_labels.append(label)

        # 随机采样
        if len(all_points) > max_points:
            sampled_indices = random.sample(range(len(all_points)), max_points)
            all_points = [all_points[i] for i in sampled_indices]
            all_labels = [all_labels[i] for i in sampled_indices]

        all_points = np.array(all_points)
        all_labels = np.array(all_labels)

        # PCA降维到二维
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_points)

        # 可视化
        plt.figure(figsize=(9, 6))
        unique_labels = sorted(set(all_labels))
        palette = sns.color_palette("Set2", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            idx = all_labels == label
            plt.scatter(reduced[idx, 0], reduced[idx, 1],
                        label=label,
                        color=color_map[label],
                        s=50, alpha=0.7, edgecolor='k', linewidth=0.3)

        plt.rcParams['font.family'] = 'SimHei'
        plt.title("Visualization of clustering", fontsize=14)
        plt.xlabel("PCA dimension 1")
        plt.ylabel("PCA dimension 2")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="类别特征", fontsize=10)
        plt.tight_layout()
        plt.savefig("../result/figures/clustering.png")
        plt.show()

    def draw_strategy_effect_comparison(self, metric_data):
        """
        可视化不同上下文策略在不同类别下的 BLEU / ROUGE-L 提升效果
        :param metric_data: dict[str, dict[str, tuple(BLEU_gain, ROUGE_gain)]]
               eg. {
                   "类别1": {"策略1": (0.02, 0.03), ...},
                   ...
               }
        """
        categories = list(metric_data.keys())
        strategies = list(next(iter(metric_data.values())).keys())
        bleu_values = []
        rouge_values = []

        # 提取数据矩阵
        for cat in categories:
            bleu_row = []
            rouge_row = []
            for strat in strategies:
                bleu_gain, rouge_gain = metric_data[cat][strat]
                bleu_row.append(bleu_gain * 100)  # 百分比展示
                rouge_row.append(rouge_gain * 100)
            bleu_values.append(bleu_row)
            rouge_values.append(rouge_row)

        x = np.arange(len(strategies))  # 策略数量
        width = 0.15  # 每组柱宽
        spacing = 0.2

        # 颜色定义
        palette = sns.color_palette("Set2", len(categories))

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, cat in enumerate(categories):
            offset = (i - 1.5) * width
            ax.bar(x + offset, bleu_values[i], width=width, label=f'{cat} - BLEU', color=palette[i], alpha=0.7)
            ax.bar(x + offset, rouge_values[i], width=width, label=f'{cat} - ROUGE-L', color=palette[i], alpha=0.4,
                   hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=10)
        ax.set_ylabel('指标提升（%）')
        ax.set_title('不同上下文策略下各类别摘要质量提升（BLEU / ROUGE-L）')
        ax.legend(ncol=2, fontsize=9)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig("../result/figures/clustering_each_strategy.png")
        plt.show()

def simulate_data():
    np.random.seed(42)
    clustering_data = {
        "类别1": np.random.normal(loc=0.2, scale=0.1, size=(200, 16)).tolist(),
        "类别2": np.random.normal(loc=0.5, scale=0.1, size=(200, 16)).tolist(),
        "类别3": np.random.normal(loc=0.8, scale=0.1, size=(200, 16)).tolist(),
        "类别4": np.random.normal(loc=1.1, scale=0.1, size=(200, 16)).tolist()
    }
    return clustering_data

def simulate_strategy_metric_data():
    categories = ["类别1", "类别2", "类别3", "类别4"]
    strategies = ["策略1", "策略2", "策略3", "策略4", "策略5"]
    data = {}
    rng = np.random.default_rng(14)

    for cat in categories:
        strat_scores = {}
        for strat in strategies:
            bleu_gain = rng.normal(loc=0.02, scale=0.015)   # 平均提升 2%
            rouge_gain = rng.normal(loc=0.03, scale=0.02)   # 平均提升 3%
            strat_scores[strat] = (bleu_gain, rouge_gain)
        data[cat] = strat_scores

    return data


if __name__ == '__main__':
    model_name = 't5-small'
    viz = Visualizer()
    viz.visualize_loss(f'../result/{model_name}_train.csv', f'../result/figures/{model_name}')

    # RQ4.1
    readability_metrics = {
        "avg_sentence_length": 12.5,
        "avg_dependency_depth": 4.8,
        "complete_sentence_ratio": 0.92
    }
    viz.visualize_readability_metrics(readability_metrics)
    viz.visualize_readability_projects()

    # RQ4.2
    accuracy_metrics = {
        "BLEU": 0.47,
        "METEOR": 0.35,
        "ROUGE-L": 0.51
    }
    viz.visualize_accuracy_metrics(accuracy_metrics)

    # RQ4.3
    # project_df = pd.read_csv("../result/project_metrics.csv")  # 包含BLEU和项目特征
    # viz.visualize_correlation_heatmap(project_df)

    # RQ5.1
    clustering_data = simulate_data()
    viz.draw_clustering2(clustering_data)

    mock_data = simulate_strategy_metric_data()
    viz.draw_strategy_effect_comparison(mock_data)
