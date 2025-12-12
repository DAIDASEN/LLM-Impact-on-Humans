import os
import numpy as np
import pandas as pd
from openai import OpenAI
from paper_clustering import PaperClusterer
import matplotlib.pyplot as plt

# 设置 API Key
API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIPaperClusterer(PaperClusterer):
    def __init__(self, csv_path, api_key, model_name='text-embedding-3-large'):
        # 调用父类初始化，但 model_name 只是个标记
        super().__init__(csv_path, model_name)
        self.client = OpenAI(api_key=api_key)
        
    def generate_embeddings(self, batch_size=20):
        """使用 OpenAI API 生成 embeddings"""
        print(f"\nGenerating embeddings using OpenAI model: {self.model_name}...")
        
        texts = self.df['combined_text'].tolist()
        # 简单的预处理：替换换行符，避免某些旧模型的问题（虽然新模型通常不需要）
        texts = [t.replace("\n", " ") for t in texts]
        
        embeddings = []
        total = len(texts)
        
        # 分批处理
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} papers)...")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                # 提取向量
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch processing: {e}")
                raise e
                
        self.embeddings = np.array(embeddings)
        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

def run_openai_analysis():
    csv_file = 'AI Impact on Humans - Sheet1.csv'
    model_name = 'text-embedding-3-large'
    
    print(f"Processing {csv_file} with {model_name}...")
    
    # 1. 初始化自定义的 OpenAI Clusterer
    clusterer = OpenAIPaperClusterer(csv_file, api_key=API_KEY, model_name=model_name)
    clusterer.load_data()
    
    # 2. 生成 Embeddings (调用 OpenAI API)
    clusterer.generate_embeddings()
    
    # 3. 自动寻找最优聚类数
    print("\n=== Finding Optimal Clusters ===")
    optimal_k = clusterer.find_optimal_clusters(max_clusters=10)
    print(f"Optimal K determined as: {optimal_k}")
    
    # 4. 定义要运行的方法 (这里只演示 K-Means，也可以加其他的)
    methods = [
        ('kmeans', 'K-Means', lambda: clusterer.cluster_kmeans(n_clusters=optimal_k)),
        ('dbscan', 'DBSCAN', lambda: clusterer.cluster_dbscan(eps=0.5, min_samples=2)), # 注意：高维向量下 DBSCAN 参数可能需要调整
        ('hierarchical', 'Hierarchical', lambda: clusterer.cluster_hierarchical(n_clusters=optimal_k))
    ]
    
    # 5. 运行并保存
    # 创建主结果目录
    safe_model_name = model_name.replace('/', '_')
    base_output_dir = f'results_{safe_model_name}'
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    for method_id, method_name, run_func in methods:
        print(f"\n{'='*20} Running {method_name} {'='*20}")
        run_func()
        
        # 子目录
        output_dir = os.path.join(base_output_dir, method_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        clusterer.analyze_clusters()
        
        # 可视化
        print(f"Generating visualizations for {method_name}...")
        for vis_method in ['pca', 'tsne', 'umap']:
            try:
                save_path = os.path.join(output_dir, f'clusters_{vis_method}.png')
                clusterer.visualize_clusters_2d(method=vis_method, save_path=save_path)
                plt.close()
            except Exception as e:
                print(f"Error generating {vis_method}: {e}")
        
        # 保存结果
        print(f"Saving results to {output_dir}...")
        clusterer.save_results(output_dir=output_dir)

if __name__ == '__main__':
    run_openai_analysis()
