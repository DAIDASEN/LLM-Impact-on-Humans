import os
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
from pathlib import Path

# 设置 API Key
# 警告：在实际项目中，建议将 API Key 放在环境变量中，而不是直接写在代码里
os.environ["OPENAI_API_KEY"] = "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"

class OpenAIClusterer:
    def __init__(self, csv_path, model_name='text-embedding-3-large'):
        self.csv_path = csv_path
        self.model_name = model_name
        self.client = OpenAI()
        self.df = None
        self.embeddings = None
        self.clusters = None

    def load_data(self):
        """加载CSV数据"""
        print(f"Loading data from {self.csv_path}...")
        encodings = ['utf-8', 'gbk', 'gb18030', 'latin1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read {self.csv_path}")
        
        self.df.columns = self.df.columns.str.lower()
        
        # 确保有 title 和 abstract
        if 'title' not in self.df.columns or 'abstract' not in self.df.columns:
             raise ValueError("CSV must contain 'title' and 'abstract' columns")

        self.df['title'] = self.df['title'].fillna('')
        self.df['abstract'] = self.df['abstract'].fillna('')
        self.df['combined_text'] = self.df['title'] + ' : ' + self.df['abstract']
        
        print(f"Loaded {len(self.df)} papers")
        return self.df

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

    def generate_embeddings(self):
        """使用 OpenAI API 生成 embeddings"""
        print(f"\nGenerating embeddings using {self.model_name} (this may take a while and cost money)...")
        
        embeddings_list = []
        total = len(self.df)
        
        for i, text in enumerate(self.df['combined_text']):
            if i % 10 == 0:
                print(f"Processing {i}/{total}...")
            try:
                emb = self.get_embedding(text)
                embeddings_list.append(emb)
            except Exception as e:
                print(f"Error at index {i}: {e}")
                # 如果出错，可以用全0向量代替或者停止
                embeddings_list.append([0]*3072) # text-embedding-3-large is 3072 dim
        
        self.embeddings = np.array(embeddings_list)
        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def find_optimal_clusters(self, max_clusters=10):
        print("\nFinding optimal number of clusters...")
        silhouettes = []
        K_range = range(2, min(max_clusters + 1, len(self.df)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            silhouettes.append(silhouette_score(self.embeddings, labels))
        
        optimal_k = K_range[np.argmax(silhouettes)]
        print(f"Recommended number of clusters: {optimal_k}")
        return optimal_k

    def cluster_kmeans(self, n_clusters):
        print(f"\nClustering with K-Means (n_clusters={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.embeddings)
        self.df['cluster'] = self.clusters
        return self.clusters

    def visualize_clusters(self, output_dir):
        print("\nVisualizing clusters...")
        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.clusters, cmap='tab10', s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'OpenAI Clusters ({self.model_name})')
        plt.savefig(os.path.join(output_dir, 'clusters_umap.png'))
        plt.close()

    def save_results(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存主文件
        cols_to_save = ['title', 'abstract', 'class', 'cluster']
        existing_cols = [col for col in cols_to_save if col in self.df.columns]
        other_cols = [c for c in self.df.columns if c not in ['title', 'abstract', 'class', 'cluster', 'combined_text']]
        final_cols = existing_cols + other_cols
        
        self.df[final_cols].to_csv(os.path.join(output_dir, 'papers_with_clusters.csv'), index=False)
        np.save(os.path.join(output_dir, 'embeddings.npy'), self.embeddings)
        
        # 分文件保存
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_papers = self.df[self.df['cluster'] == cluster_id]
            cluster_papers[final_cols].to_csv(os.path.join(output_dir, f'cluster_{cluster_id}.csv'), index=False)
            
        print(f"Results saved to {output_dir}")

def main():
    csv_file = 'AI Impact on Humans - Sheet1.csv'
    model_name = 'text-embedding-3-large'
    
    clusterer = OpenAIClusterer(csv_file, model_name=model_name)
    clusterer.load_data()
    clusterer.generate_embeddings()
    
    optimal_k = clusterer.find_optimal_clusters()
    clusterer.cluster_kmeans(n_clusters=optimal_k)
    
    output_dir = f'results_{model_name}'
    clusterer.visualize_clusters(output_dir)
    clusterer.save_results(output_dir)

if __name__ == '__main__':
    main()
