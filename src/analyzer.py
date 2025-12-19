import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class LayerAnalyzer:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()

    def compute_similarity(self, text: str) -> np.ndarray:
        """
        入力テキストに対する各層の隠れ状態を取得し、層間のコサイン類似度行列を計算する
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # hidden_states: タプル (入力embeddings + 各層の出力)
        # 形状: (layer_count, batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states
        
        # 各層の最終トークンのベクトルのみを抽出して代表値とする（簡易実装）
        # shape: (num_layers, hidden_dim)
        layer_vectors = [h[0, -1, :].cpu().numpy() for h in hidden_states]
        
        # コサイン類似度行列の計算 (num_layers x num_layers)
        sim_matrix = cosine_similarity(layer_vectors)
        
        return sim_matrix
