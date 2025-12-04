import os
from paper_clustering import PaperClusterer
import matplotlib.pyplot as plt

def run_comprehensive_analysis():
    # 1. 初始化
    csv_file = 'AI Impact on Humans - Sheet1.csv'
    print(f"Processing {csv_file}...")
    
    # 切换为更高质量的模型
    # 选项 A: 'all-MiniLM-L6-v2' (快速，默认)
    # 选项 B: 'all-mpnet-base-v2' (高质量，推荐，768维)
    # 选项 C: 'allenai-specter' (专门针对学术论文)
    # 选项 D: 'BAAI/bge-large-en-v1.5' (目前开源最强之一，需下载约1.3GB)
    
    model_name = 'all-mpnet-base-v2' 
    print(f"Using model: {model_name}")
    
    clusterer = PaperClusterer(csv_file, model_name=model_name)
    clusterer.load_data()
    clusterer.generate_embeddings()
    
    # 2. 自动寻找最优聚类数 (为 K-Means 和 Hierarchical 使用)
    print("\n=== Finding Optimal Clusters ===")
    optimal_k = clusterer.find_optimal_clusters(max_clusters=10)
    print(f"Optimal K determined as: {optimal_k}")
    
    # 定义要运行的方法
    methods = [
        ('kmeans', 'K-Means', lambda: clusterer.cluster_kmeans(n_clusters=optimal_k)),
        ('dbscan', 'DBSCAN', lambda: clusterer.cluster_dbscan(eps=0.5, min_samples=2)),
        ('hierarchical', 'Hierarchical', lambda: clusterer.cluster_hierarchical(n_clusters=optimal_k))
    ]
    
    # 3. 循环执行每种方法
    for method_id, method_name, run_func in methods:
        print(f"\n{'='*20} Running {method_name} {'='*20}")
        
        # 运行聚类
        run_func()
        
        # 创建输出目录
        # 使用模型名称作为主文件夹，避免覆盖
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        output_dir = os.path.join(f'results_{safe_model_name}', method_id)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 分析
        clusterer.analyze_clusters()
        
        # 可视化 (PCA, t-SNE, UMAP)
        print(f"Generating visualizations for {method_name}...")
        vis_methods = ['pca', 'tsne', 'umap']
        for vis_method in vis_methods:
            try:
                save_path = os.path.join(output_dir, f'clusters_{vis_method}.png')
                clusterer.visualize_clusters_2d(method=vis_method, save_path=save_path)
                plt.close() # 关闭图表释放内存
            except Exception as e:
                print(f"Error generating {vis_method} visualization: {e}")
        
        # 保存结果 (包含 Class 列)
        print(f"Saving results to {output_dir}...")
        clusterer.save_results(output_dir=output_dir)
        
    print("\nAll analyses completed successfully!")

if __name__ == '__main__':
    run_comprehensive_analysis()
