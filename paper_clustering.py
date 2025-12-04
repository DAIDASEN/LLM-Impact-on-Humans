# paper_clustering.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score  # 添加这行
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class PaperClusterer:
    def __init__(self, csv_path, model_name='all-MiniLM-L6-v2'):
        """
        初始化论文聚类器
        
        Args:
            csv_path: CSV文件路径，需包含 'title' 和 'abstract' 列
            model_name: 使用的embedding模型
                推荐模型：
                - 'all-MiniLM-L6-v2': 快速，384维 (推荐)
                - 'all-mpnet-base-v2': 高质量，768维
                - 'allenai-specter': 专门为科学论文设计
        """
        self.csv_path = csv_path
        self.model_name = model_name
        self.df = None
        self.embeddings = None
        self.model = None
        self.clusters = None
        
    def load_data(self):
        """加载CSV数据"""
        print(f"Loading data from {self.csv_path}...")
        # 尝试多种编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read {self.csv_path} with any of these encodings: {encodings}")
        
        # 标准化列名（转为小写）
        self.df.columns = self.df.columns.str.lower()
        
        # 检查必需的列
        required_cols = ['title', 'abstract']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}. Available columns: {list(self.df.columns)}")
        
        # 处理缺失值
        self.df['title'] = self.df['title'].fillna('')
        self.df['abstract'] = self.df['abstract'].fillna('')
        
        # 合并标题和摘要
        self.df['combined_text'] = self.df['title'] + ' [SEP] ' + self.df['abstract']
        
        print(f"Loaded {len(self.df)} papers")
        return self.df
    
    def generate_embeddings(self, batch_size=32):
        """生成文本embeddings"""
        print(f"\nGenerating embeddings using {self.model_name}...")
        
        # 加载模型
        self.model = SentenceTransformer(self.model_name)
        
        # 生成embeddings
        texts = self.df['combined_text'].tolist()
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_kmeans(self, n_clusters=5, random_state=42):
        """使用K-Means聚类"""
        print(f"\nClustering with K-Means (n_clusters={n_clusters})...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        self.clusters = kmeans.fit_predict(self.embeddings)
        self.df['cluster'] = self.clusters
        
        # 计算聚类质量指标
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        silhouette = silhouette_score(self.embeddings, self.clusters)
        calinski = calinski_harabasz_score(self.embeddings, self.clusters)
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski:.4f}")
        
        return self.clusters
    
    def cluster_dbscan(self, eps=0.5, min_samples=2):
        """使用DBSCAN聚类（自动确定聚类数）"""
        print(f"\nClustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        self.clusters = dbscan.fit_predict(self.embeddings)
        self.df['cluster'] = self.clusters
        
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        n_noise = list(self.clusters).count(-1)
        
        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise}")
        
        return self.clusters
    
    def cluster_hierarchical(self, n_clusters=5, linkage='ward'):
        """使用层次聚类"""
        print(f"\nClustering with Hierarchical (n_clusters={n_clusters}, linkage={linkage})...")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        self.clusters = hierarchical.fit_predict(self.embeddings)
        self.df['cluster'] = self.clusters
        
        return self.clusters
    
    def find_optimal_clusters(self, max_clusters=10):
        """使用肘部法则找最优聚类数"""
        print("\nFinding optimal number of clusters...")
        
        inertias = []
        silhouettes = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.embeddings, labels))
        
        # 绘制肘部曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(K_range, silhouettes, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        print("Saved: optimal_clusters.png")
        
        # 推荐最优k（基于silhouette score）
        optimal_k = K_range[np.argmax(silhouettes)]
        print(f"\nRecommended number of clusters: {optimal_k}")
        
        return optimal_k
    
    def visualize_clusters_2d(self, method='umap', save_path='clusters_2d.png'):
        """2D可视化聚类结果"""
        print(f"\nVisualizing clusters using {method.upper()}...")
        
        # 降维到2D
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(self.embeddings)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(self.embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(self.embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 绘图
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=self.clusters,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Paper Clusters ({method.upper()} visualization)')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        return embeddings_2d
    
    def analyze_clusters(self, top_n=5):
        """分析每个聚类的特征"""
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS")
        print("="*80)
        
        cluster_info = []
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            if cluster_id == -1:  # DBSCAN的噪声点
                continue
            
            cluster_papers = self.df[self.df['cluster'] == cluster_id]
            n_papers = len(cluster_papers)
            
            print(f"\n{'='*80}")
            print(f"Cluster {cluster_id} ({n_papers} papers)")
            print(f"{'='*80}")
            
            # 显示代表性论文（最接近聚类中心）
            cluster_embeddings = self.embeddings[self.df['cluster'] == cluster_id]
            cluster_center = cluster_embeddings.mean(axis=0)
            
            # 计算到中心的距离
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            top_indices = np.argsort(distances)[:top_n]
            
            print(f"\nTop {top_n} representative papers:")
            for i, idx in enumerate(top_indices, 1):
                paper = cluster_papers.iloc[idx]
                print(f"\n{i}. {paper['title']}")
                print(f"   Abstract: {paper['abstract'][:200]}...")
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'n_papers': n_papers,
                'percentage': f"{n_papers/len(self.df)*100:.1f}%"
            })
        
        # 聚类大小分布
        print(f"\n{'='*80}")
        print("CLUSTER SIZE DISTRIBUTION")
        print(f"{'='*80}")
        cluster_df = pd.DataFrame(cluster_info)
        print(cluster_df.to_string(index=False))
        
        return cluster_info
    
    def save_results(self, output_dir='clustering_results'):
        """保存聚类结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存带聚类标签的CSV
        output_csv = output_dir / 'papers_with_clusters.csv'
        self.df.to_csv(output_csv, index=False)
        print(f"\nSaved clustered papers to: {output_csv}")
        
        # 保存embeddings
        embeddings_path = output_dir / 'embeddings.npy'
        np.save(embeddings_path, self.embeddings)
        print(f"Saved embeddings to: {embeddings_path}")
        
        # 保存每个聚类的论文
        for cluster_id in sorted(self.df['cluster'].unique()):
            if cluster_id == -1:
                continue
            cluster_papers = self.df[self.df['cluster'] == cluster_id]
            cluster_file = output_dir / f'cluster_{cluster_id}.csv'
            
            # 尝试保存更多有用的列，如果存在
            cols_to_save = ['title', 'abstract', 'class', 'cluster']
            # 过滤出实际存在的列
            existing_cols = [col for col in cols_to_save if col in self.df.columns]
            # 如果有其他列想保留，也可以直接保存所有列：
            # cluster_papers.to_csv(cluster_file, index=False)
            # 这里我们优先保存指定的列，如果class存在的话
            
            if 'class' in self.df.columns:
                 # 如果有class列，优先展示
                 other_cols = [c for c in self.df.columns if c not in ['title', 'abstract', 'class', 'cluster', 'combined_text']]
                 final_cols = existing_cols + other_cols
                 cluster_papers[final_cols].to_csv(cluster_file, index=False)
            else:
                 cluster_papers.to_csv(cluster_file, index=False)
        
        print(f"\nAll results saved to: {output_dir}")
        
        return output_dir


def main():
    """主函数示例"""
    
    # 1. 初始化聚类器
    clusterer = PaperClusterer(
        csv_path='papers.csv',
        model_name='all-MiniLM-L6-v2'  # 或使用 'allenai-specter' 专门处理科学论文
    )
    
    # 2. 加载数据
    clusterer.load_data()
    
    # 3. 生成embeddings
    clusterer.generate_embeddings()
    
    # 4. 找最优聚类数（可选）
    optimal_k = clusterer.find_optimal_clusters(max_clusters=10)
    
    # 5. 执行聚类
    clusterer.cluster_kmeans(n_clusters=optimal_k)
    # 或者使用其他方法：
    # clusterer.cluster_dbscan(eps=0.5, min_samples=2)
    # clusterer.cluster_hierarchical(n_clusters=5)
    
    # 6. 可视化
    clusterer.visualize_clusters_2d(method='umap', save_path='clusters_umap.png')
    clusterer.visualize_clusters_2d(method='tsne', save_path='clusters_tsne.png')
    
    # 7. 分析聚类
    clusterer.analyze_clusters(top_n=5)
    
    # 8. 保存结果
    clusterer.save_results(output_dir='clustering_results')


if __name__ == '__main__':
    main()
