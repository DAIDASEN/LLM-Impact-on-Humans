# quick_cluster.py
from paper_clustering import PaperClusterer

# 一键聚类
clusterer = PaperClusterer('papers.csv', model_name='all-MiniLM-L6-v2')
clusterer.load_data()
clusterer.generate_embeddings()

# 手动设定聚类数为 4
n_clusters = 3

# 聚类
clusterer.cluster_kmeans(n_clusters=n_clusters)

# 可视化和分析
clusterer.visualize_clusters_2d(method='umap')
clusterer.analyze_clusters()
clusterer.save_results()
