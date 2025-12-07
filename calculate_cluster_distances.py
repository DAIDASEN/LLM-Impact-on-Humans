"""
聚类间距离计算脚本
用于分析 results_all-mpnet-base-v2/kmeans 中的聚类中心距离
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path

def calculate_cluster_distances():
    # 1. 加载数据
    print("=" * 80)
    print("聚类间距离分析")
    print("=" * 80)
    
    results_dir = Path('results_all-mpnet-base-v2/kmeans')
    
    if not results_dir.exists():
        print(f"❌ 错误：找不到路径 {results_dir}")
        print("请确保你在包含 results_all-mpnet-base-v2 文件夹的目录下运行此脚本")
        return
    
    print(f"\n✓ 已加载结果目录: {results_dir}")
    
    # 加载 embeddings
    embeddings = np.load(results_dir / 'embeddings.npy')
    print(f"✓ Embeddings 形状: {embeddings.shape} (44 篇论文, 768 维)")
    
    # 加载聚类标签
    df = pd.read_csv(results_dir / 'papers_with_clusters.csv')
    cluster_labels = df['cluster'].values
    
    # 2. 计算聚类中心
    n_clusters = len(np.unique(cluster_labels))
    centroids = []
    
    for i in range(n_clusters):
        cluster_points = embeddings[cluster_labels == i]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
        print(f"✓ Cluster {i}: {len(cluster_points)} 篇论文")
    
    centroids = np.array(centroids)
    
    # 3. 计算各种距离矩阵
    print("\n" + "=" * 80)
    print("1. 聚类中心之间的欧几里得距离矩阵")
    print("=" * 80)
    
    euclidean_dist = euclidean_distances(centroids)
    euclidean_df = pd.DataFrame(
        euclidean_dist,
        index=[f'Cluster {i}' for i in range(n_clusters)],
        columns=[f'Cluster {i}' for i in range(n_clusters)]
    )
    print(euclidean_df.round(4))
    
    print("\n" + "=" * 80)
    print("2. 聚类中心之间的余弦距离矩阵")
    print("=" * 80)
    
    cosine_dist = cosine_distances(centroids)
    cosine_df = pd.DataFrame(
        cosine_dist,
        index=[f'Cluster {i}' for i in range(n_clusters)],
        columns=[f'Cluster {i}' for i in range(n_clusters)]
    )
    print(cosine_df.round(4))
    
    print("\n" + "=" * 80)
    print("3. 聚类中心之间的余弦相似度")
    print("=" * 80)
    
    cosine_sim = cosine_similarity(centroids)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=[f'Cluster {i}' for i in range(n_clusters)],
        columns=[f'Cluster {i}' for i in range(n_clusters)]
    )
    print(cosine_sim_df.round(4))
    
    # 4. 聚类对距离统计
    print("\n" + "=" * 80)
    print("4. 聚类对距离统计（按欧几里得距离排序）")
    print("=" * 80)
    
    distances_list = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            distances_list.append({
                'Cluster_Pair': f'Cluster {i} - Cluster {j}',
                'Euclidean_Distance': euclidean_dist[i, j],
                'Cosine_Distance': cosine_dist[i, j],
                'Cosine_Similarity': cosine_sim[i, j]
            })
    
    distances_df = pd.DataFrame(distances_list)
    distances_df = distances_df.sort_values('Euclidean_Distance', ascending=False)
    
    print(distances_df.to_string(index=False))
    
    print("\n最远的聚类对（欧几里得距离）：")
    print(f"  {distances_df.iloc[0]['Cluster_Pair']}: {distances_df.iloc[0]['Euclidean_Distance']:.4f}")
    
    print("\n最近的聚类对（欧几里得距离）：")
    print(f"  {distances_df.iloc[-1]['Cluster_Pair']}: {distances_df.iloc[-1]['Euclidean_Distance']:.4f}")
    
    # 5. 聚类质量评估
    print("\n" + "=" * 80)
    print("5. 聚类质量评估指标")
    print("=" * 80)
    
    silhouette = silhouette_score(embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
    
    print(f"\n✓ Silhouette Score (范围: -1 到 1, 越高越好): {silhouette:.4f}")
    print(f"  解释: ", end="")
    if silhouette > 0.5:
        print("聚类效果较好")
    elif silhouette > 0:
        print("聚类效果一般")
    else:
        print("聚类效果较差")
    
    print(f"\n✓ Davies-Bouldin Index (越低越好): {davies_bouldin:.4f}")
    print(f"  解释: 表示簇内距离/簇间距离的平均比值，范围通常为 0-3")
    
    print(f"\n✓ Calinski-Harabasz Index (越高越好): {calinski_harabasz:.4f}")
    print(f"  解释: 簇间方差/簇内方差的比值，值越高说明聚类效果越好")
    
    # 6. 聚类内部紧密度
    print("\n" + "=" * 80)
    print("6. 每个聚类内部的紧密度（到中心的平均距离）")
    print("=" * 80)
    
    cluster_cohesion = []
    for i in range(n_clusters):
        cluster_points = embeddings[cluster_labels == i]
        centroid = centroids[i]
        distances_to_center = euclidean_distances([centroid], cluster_points)[0]
        avg_distance = distances_to_center.mean()
        cluster_cohesion.append({
            'Cluster': f'Cluster {i}',
            'Papers': len(cluster_points),
            'Avg_Distance_to_Center': avg_distance,
            'Max_Distance_to_Center': distances_to_center.max(),
            'Min_Distance_to_Center': distances_to_center.min()
        })
    
    cohesion_df = pd.DataFrame(cluster_cohesion)
    print(cohesion_df.to_string(index=False))
    
    # 7. 保存所有结果到 CSV
    print("\n" + "=" * 80)
    print("7. 保存结果")
    print("=" * 80)
    
    euclidean_df.to_csv('cluster_distances_euclidean.csv')
    print("✓ 已保存: cluster_distances_euclidean.csv")
    
    cosine_df.to_csv('cluster_distances_cosine.csv')
    print("✓ 已保存: cluster_distances_cosine.csv")
    
    cosine_sim_df.to_csv('cluster_similarity_cosine.csv')
    print("✓ 已保存: cluster_similarity_cosine.csv")
    
    distances_df.to_csv('cluster_pairs_distances.csv', index=False)
    print("✓ 已保存: cluster_pairs_distances.csv")
    
    cohesion_df.to_csv('cluster_cohesion.csv', index=False)
    print("✓ 已保存: cluster_cohesion.csv")
    
    # 8. 总结
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)
    
    print(f"""
✓ 使用模型: all-mpnet-base-v2 (768维)
✓ 聚类数: {n_clusters}
✓ 论文总数: {len(embeddings)}

关键发现:
1. 最相近的两个聚类: {distances_df.iloc[-1]['Cluster_Pair']} 
   (欧几里得距离: {distances_df.iloc[-1]['Euclidean_Distance']:.4f})

2. 最远的两个聚类: {distances_df.iloc[0]['Cluster_Pair']} 
   (欧几里得距离: {distances_df.iloc[0]['Euclidean_Distance']:.4f})

3. 聚类质量评估:
   - Silhouette Score: {silhouette:.4f}
   - Davies-Bouldin Index: {davies_bouldin:.4f}
   - Calinski-Harabasz Index: {calinski_harabasz:.4f}

建议: 
- 如果 Silhouette Score 较低 (< 0.3)，可以尝试调整聚类数
- 如果想要更明显的聚类分离，可以考虑使用维数更高的模型
""")
    
    print("=" * 80)
    print("✓ 分析完成！所有结果已保存。")
    print("=" * 80)

if __name__ == '__main__':
    calculate_cluster_distances()
