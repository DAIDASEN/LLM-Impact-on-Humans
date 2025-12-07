# 聚类间距离分析报告

## 1. 数据来源与模型信息

| 项目 | 详情 |
|------|------|
| **Embedding 模型** | `all-mpnet-base-v2` |
| **模型维度** | 768 维 |
| **聚类算法** | K-Means |
| **自动推荐聚类数** | 4 类 |
| **数据集** | AI Impact on Humans - Sheet1.csv (44篇论文) |

---

## 2. 聚类分布

| 聚类ID | 论文数 | 占比 | 主题 |
|--------|-------|------|------|
| Cluster 0 | 9篇 | 20.5% | AI 伴侣与情感支持 |
| Cluster 1 | 13篇 | 29.5% | 政治说服与决策 |
| Cluster 2 | 9篇 | 20.5% | 创意与学习 |
| Cluster 3 | 13篇 | 29.5% | 教育与认知 |

---

## 3. 类与类之间的距离计算方法

### 3.1 概念说明

在 K-Means 聚类中，**类与类之间的距离**通常指的是**聚类中心（Centroid）之间的距离**。由于 Embedding 是 768 维的向量，可以用以下几种距离度量：

#### (1) **欧几里得距离（Euclidean Distance）** ⭐ 最常用
$$d_{euclidean} = \sqrt{\sum_{i=1}^{768} (x_i - y_i)^2}$$

#### (2) **余弦相似度（Cosine Similarity）**
$$cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$
- 范围：[-1, 1]，1 表示完全相同，-1 表示完全相反

#### (3) **曼哈顿距离（Manhattan Distance）**
$$d_{manhattan} = \sum_{i=1}^{768} |x_i - y_i|$$

#### (4) **余弦距离（Cosine Distance）**
$$d_{cosine} = 1 - cos(\theta)$$

---

## 4. 计算聚类中心距离的 Python 代码

### 方法一：使用现有的聚类对象（推荐）

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pathlib import Path

# 1. 加载聚类结果
results_dir = Path('results_all-mpnet-base-v2/kmeans')

# 加载 embeddings
embeddings = np.load(results_dir / 'embeddings.npy')  # 形状: (44, 768)

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

centroids = np.array(centroids)

# 3. 计算类与类之间的距离
print("=" * 60)
print("聚类中心之间的欧几里得距离矩阵")
print("=" * 60)

euclidean_dist = euclidean_distances(centroids)
euclidean_df = pd.DataFrame(
    euclidean_dist,
    index=[f'Cluster {i}' for i in range(n_clusters)],
    columns=[f'Cluster {i}' for i in range(n_clusters)]
)
print(euclidean_df.round(4))

print("\n" + "=" * 60)
print("聚类中心之间的余弦距离矩阵")
print("=" * 60)

cosine_dist = cosine_distances(centroids)
cosine_df = pd.DataFrame(
    cosine_dist,
    index=[f'Cluster {i}' for i in range(n_clusters)],
    columns=[f'Cluster {i}' for i in range(n_clusters)]
)
print(cosine_df.round(4))

print("\n" + "=" * 60)
print("聚类中心之间的余弦相似度")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(centroids)
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=[f'Cluster {i}' for i in range(n_clusters)],
    columns=[f'Cluster {i}' for i in range(n_clusters)]
)
print(cosine_sim_df.round(4))

# 4. 找出最相近和最远的聚类对
print("\n" + "=" * 60)
print("聚类对距离统计")
print("=" * 60)

distances_list = []
for i in range(n_clusters):
    for j in range(i+1, n_clusters):
        distances_list.append({
            'Cluster Pair': f'Cluster {i} - Cluster {j}',
            'Euclidean Distance': euclidean_dist[i, j],
            'Cosine Distance': cosine_dist[i, j],
            'Cosine Similarity': cosine_sim[i, j]
        })

distances_df = pd.DataFrame(distances_list)
distances_df = distances_df.sort_values('Euclidean Distance', ascending=False)

print("\n最远的聚类对（欧几里得距离）：")
print(distances_df.iloc[0])

print("\n最近的聚类对（欧几里得距离）：")
print(distances_df.iloc[-1])

# 5. 保存结果到 CSV
distances_df.to_csv('cluster_distances.csv', index=False)
print(f"\n结果已保存到 cluster_distances.csv")
```

### 方法二：直接从聚类器对象获取（如果你还有原始代码）

```python
from paper_clustering import PaperClusterer

# 1. 初始化并加载
clusterer = PaperClusterer('AI Impact on Humans - Sheet1.csv', model_name='all-mpnet-base-v2')
clusterer.load_data()
clusterer.generate_embeddings()
clusterer.cluster_kmeans(n_clusters=4)

# 2. 获取聚类中心
kmeans_model = clusterer.kmeans  # 或通过重新运行 KMeans 得到
centroids = kmeans_model.cluster_centers_

# 3. 后续步骤同方法一的第 3-5 部分
```

---

## 5. 预期输出示例

### 欧几里得距离矩阵
```
           Cluster 0  Cluster 1  Cluster 2  Cluster 3
Cluster 0      0.0000    15.3421    12.8504    14.2156
Cluster 1     15.3421     0.0000    18.5632    11.2341
Cluster 2     12.8504    18.5632     0.0000    13.9876
Cluster 3     14.2156    11.2341    13.9876     0.0000
```

### 聚类对距离统计
```
        Cluster Pair  Euclidean Distance  Cosine Distance  Cosine Similarity
0  Cluster 1 - Cluster 2           18.5632           0.3421           0.6579
1  Cluster 0 - Cluster 1           15.3421           0.2891           0.7109
2  Cluster 3 - Cluster 0           14.2156           0.2654           0.7346
3  Cluster 1 - Cluster 3           11.2341           0.1987           0.8013
```

---

## 6. 解读距离指标

### 欧几里得距离
- **值越小**：两个聚类在语义空间上**越接近**
- **适用场景**：一般性的相似度衡量

### 余弦距离 / 余弦相似度
- **余弦距离越小**（或余弦相似度越大）：聚类方向**越接近**
- **对量级不敏感**：只关注方向，不关注幅度
- **适用场景**：文本向量、方向相似度

### 聚类紧密度指标
- 如果聚类中心距离都较小（如 < 5），说明**聚类分化不明显**，可能需要减少聚类数
- 如果距离相差很大（如 5 - 30），说明**聚类分化明显**，聚类效果较好

---

## 7. 进阶分析：聚类质量评估

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 计算聚类评估指标
silhouette = silhouette_score(embeddings, cluster_labels)
davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)

print(f"Silhouette Score（-1 到 1，越高越好）: {silhouette:.4f}")
print(f"Davies-Bouldin Index（越低越好）: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index（越高越好）: {calinski_harabasz:.4f}")
```

| 指标 | 含义 | 最优值 |
|------|------|--------|
| **Silhouette Score** | 聚类紧密度与分离度 | 接近 1 |
| **Davies-Bouldin Index** | 簇内距/簇间距的比值 | 越低越好 |
| **Calinski-Harabasz Index** | 簇间方差/簇内方差 | 越高越好 |

---

## 8. 快速运行脚本

将以下代码保存为 `calculate_cluster_distances.py`，直接运行：

```bash
python calculate_cluster_distances.py
```

脚本会自动：
1. 加载 `results_all-mpnet-base-v2/kmeans` 的数据
2. 计算所有距离矩阵
3. 输出结果并保存为 CSV 文件

---

## 9. 总结

- **模型**：all-mpnet-base-v2（768维）
- **聚类数**：4 类
- **距离计算**：推荐使用**欧几里得距离**或**余弦相似度**
- **应用场景**：
  - 比较聚类的相似性
  - 评估聚类质量
  - 决定是否需要调整聚类参数

