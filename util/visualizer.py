import matplotlib.pyplot as plt
import pandas as pd
import os

class Visualizer:
    def __init__(self, csv_path="../result/train.csv"):
        self.csv_path = csv_path

    def loss_visualize(self, save_path=None):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")

        # 加载数据
        df = pd.read_csv(self.csv_path)

        if 'train_loss' not in df.columns or 'val_loss' not in df.columns:
            raise ValueError("CSV 中必须包含 'train_loss' 和 'val_loss' 列")

        # 开始绘图
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2, color='#1f77b4')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')

        # 图像美化
        plt.title("Training and Validation Loss over Epochs", fontsize=16, fontweight='bold')
        plt.xlabel("Epoch", fontsize=13)
        plt.ylabel("Loss", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(df['epoch'])
        plt.legend()
        plt.tight_layout()

        # 保存或展示
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Loss curve saved to {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    model_name = 't5-small'
    Visualizer(f'../result/{model_name}_train.csv').loss_visualize(f'../result/figures/{model_name}.png')
