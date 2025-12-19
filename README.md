# DFS Layer Visualizer

A visualization tool for analyzing the **Data Flow Space (DFS)** of Large Language Models (LLMs).
This tool computes and visualizes the cosine similarity between hidden states of different layers, helping researchers understand layer redundancy and optimize **Model Merging** strategies (e.g., Layer Swapping/Permutation).

## Features
- **Hidden State Extraction**: Automatically hooks into PyTorch/HuggingFace models to capture layer outputs.
- **DFS Similarity Matrix**: Computes cosine similarity between all layers to identify the "Data Flow" consistency.
- **Visualization**: Generates clear heatmaps to identify swap-compatible layers.

## Installation
```bash
git clone [https://github.com/YourName/dfs-layer-visualizer.git](https://github.com/YourName/dfs-layer-visualizer.git)
cd dfs-layer-visualizer
pip install -r requirements.txt
```

## Usage
```python
from src.analyzer import LayerAnalyzer
from src.visualizer import plot_heatmap

# Load your model (e.g., GPT-2, Llama-3, etc.)
model_name = "gpt2"
analyzer = LayerAnalyzer(model_name)

# Calculate similarity matrix on a sample input
similarity_matrix = analyzer.compute_similarity("Hello, world! This is a test for DFS.")

# Visualize
plot_heatmap(similarity_matrix, title="GPT-2 Layer Similarity in Data Flow Space")
```
<img width="1118" height="1007" alt="image" src="https://github.com/user-attachments/assets/813fab68-27a7-402b-b9d1-3e94e0f37191" />

