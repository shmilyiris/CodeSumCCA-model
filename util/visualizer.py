import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns


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

        # Save the figure
        plt.tight_layout()
        plt.savefig('../result/figures/readability_metrics_projects.png')

        # Show the plot
        plt.show()


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
    project_df = pd.read_csv("../result/project_metrics.csv")  # 包含BLEU和项目特征
    viz.visualize_correlation_heatmap(project_df)
