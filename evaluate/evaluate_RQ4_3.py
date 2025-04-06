import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class CorrelationEvaluation:
    def __init__(self, project_metrics: list[dict]):
        """
        project_metrics: List of project dicts with fields:
            - project_name
            - class_count
            - method_avg
            - field_avg
            - avg_class_length
            - avg_dependency_count
            - avg_bleu
        """
        self.df = pd.DataFrame(project_metrics)

    def compute_correlation(self):
        correlations = {}
        for feature in ['class_count', 'method_avg', 'field_avg', 'avg_class_length', 'avg_dependency_count']:
            corr, _ = pearsonr(self.df[feature], self.df['avg_bleu'])
            correlations[feature] = round(corr, 4)
        return correlations

    def visualize_correlation(self):
        import seaborn as sns
        import matplotlib.pyplot as plt

        corr_matrix = self.df.corr(numeric_only=True)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()
