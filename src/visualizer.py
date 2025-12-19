import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(matrix: np.ndarray, title: str = "Layer Similarity", save_path: str = "heatmap.png"):
    """
    類似度行列をヒートマップとして描画・保存する
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix, 
        annot=False, 
        cmap="viridis", 
        square=True,
        xticklabels=5, 
        yticklabels=5
    )
    plt.title(title)
    plt.xlabel("Layer Index (Target)")
    plt.ylabel("Layer Index (Source)")
    plt.tight_layout()
    
    print(f"Saving heatmap to {save_path}...")
    plt.savefig(save_path, dpi=300)
    plt.show()
