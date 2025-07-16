import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

from constants import NUM_CLASTERS


def cluster_embeddings(embeddings_file: Path, output_dir: Path, n_clusters: int = 10) -> pd.DataFrame:
    """Кластеризует эмбеддинги и визуализирует результаты
    
    Parameters
    ----------
    embeddings_file : Path
        путь к файлу с эмбеддингами 
    output_dir : Path
        директория для сохранения результатов
    n_clusters : int, optional
        количество кластеров , по умолчанию 10

    Returns
    -------
    pd.DataFrame
        DataFrame с информацией о кластерах
    """
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    
    video_ids = list(embeddings_data.keys())
    embeddings = np.array([embeddings_data[vid] for vid in video_ids])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Оценка качества кластеризации
    silhouette = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    cluster_df = pd.DataFrame({
        'video_id': video_ids,
        'cluster_id': clusters
    })
    cluster_df.to_csv(output_dir / 'video_cluster_mapping.csv', index=False)
    
    with open(output_dir / 'clustering_metrics.txt', 'w') as f:
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Davies-Bouldin Index: {davies_bouldin:.4f}\n")
    
    # Визуализация с помощью t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=clusters, 
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('t-SNE Visualization of Video Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(output_dir / 'tsne_visualization.png')
    
    # Распределение по кластерам
    plt.figure(figsize=(10, 6))
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
    plt.title('Video Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Videos')
    plt.savefig(output_dir / 'cluster_distribution.png')
    
    return cluster_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--n_clusters', type=int, default=NUM_CLASTERS)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cluster_embeddings(
        embeddings_file=Path(args.embeddings_file),
        output_dir=output_dir,
        n_clusters=args.n_clusters
    )