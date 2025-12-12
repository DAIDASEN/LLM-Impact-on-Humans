import time
import random
import pandas as pd
import numpy as np
from scholarly import scholarly, ProxyGenerator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================

# 文件路径 (请确保这些文件在当前目录下)
EXISTING_CSV_PATH = 'papers_with_clusters.csv'
EXISTING_EMBEDDINGS_PATH = 'embeddings.npy'
OUTPUT_CSV_PATH = 'new_classified_papers.csv'

# Embedding 模型 (必须与生成 .npy 时使用的模型一致)
MODEL_NAME = 'all-mpnet-base-v2'

# 相似度阈值 (低于此值被视为无关论文)
SIMILARITY_THRESHOLD = 0.35

# 针对四个领域的搜索关键词 (逻辑: 核心词 AND 领域词 AND 细分词)
SEARCH_QUERIES = {
    0: '("Large Language Models" OR "AI Agents") AND ("Human-AI Collaboration" OR "Teaming") AND ("Companionship" OR "Anthropomorphism")',
    1: '("Large Language Models" OR "ChatGPT") AND ("Psychology" OR "Mental Health") AND ("Persuasion" OR "Belief Change")',
    2: '("Large Language Models" OR "Generative AI") AND ("Creativity" OR "Ideation") AND ("Divergent Thinking" OR "Novelty")',
    3: '("Large Language Models" OR "ChatGPT") AND ("Education" OR "Productivity") AND ("Writing Skills" OR "Learning Outcomes")'
}

CLUSTER_NAMES = {
    0: "Cluster 0: Human-AI Collaboration",
    1: "Cluster 1: Psychology & Persuasion",
    2: "Cluster 2: Creativity & Ideation",
    3: "Cluster 3: Education & Productivity"
}

# ==========================================
# 2. 核心类定义
# ==========================================

class PaperClassifier:
    def __init__(self, csv_path, npy_path, model_name):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("Loading existing data and calculating centroids...")
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(npy_path)
        self.centroids = self._calculate_centroids()
        print("Classifier initialized successfully.")

    def _calculate_centroids(self):
        centroids = {}
        for cluster_id in range(4):
            # 获取属于该聚类的所有索引
            indices = self.df[self.df['cluster'] == cluster_id].index
            if len(indices) == 0:
                print(f"Warning: Cluster {cluster_id} has no data.")
                continue
            # 计算平均向量 (Centroid)
            cluster_embeddings = self.embeddings[indices]
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
        return centroids

    def classify(self, title, abstract):
        """
        输入标题和摘要，返回 (预测类别ID, 相似度得分, 详细距离信息)
        如果得分低于阈值，预测类别ID 为 -1 (Unrelated)
        """
        # 构造输入文本，保持格式一致性
        text = f"{title} [SEP] {abstract}"
        vector = self.model.encode([text])
        
        scores = {}
        for cid, centroid in self.centroids.items():
            # 计算余弦相似度
            sim = cosine_similarity(vector, centroid.reshape(1, -1))[0][0]
            scores[cid] = sim
            
        # 找到最相似的 Cluster
        best_cluster = max(scores, key=scores.get)
        best_score = scores[best_cluster]
        
        if best_score < SIMILARITY_THRESHOLD:
            return -1, best_score, scores # -1 表示无关
        else:
            return best_cluster, best_score, scores

# ==========================================
# 3. 爬虫函数
# ==========================================

def fetch_papers_from_google_scholar(queries, max_per_query=5):
    """
    爬取指定 Query 的论文。
    注意：Google Scholar 有严格的反爬限制。
    如果遇到 CAPTCHA，程序可能会报错或需要人工干预。
    """
    fetched_data = []
    
    # 可选：设置代理 (如果本地无法直连 Google Scholar)
    # pg = ProxyGenerator()
    # pg.FreeProxies()
    # scholarly.use_proxy(pg)

    for cluster_id, query in queries.items():
        print(f"\nSearching for Cluster {cluster_id}: {query[:50]}...")
        try:
            search_query = scholarly.search_pubs(query)
            count = 0
            
            for pub in search_query:
                if count >= max_per_query:
                    break
                
                # 提取必要信息
                # bib 包含 title, abstract, author, venue, year 等
                bib = pub.get('bib', {})
                title = bib.get('title', 'N/A')
                abstract = bib.get('abstract', 'N/A')
                pub_url = pub.get('pub_url', 'N/A')
                year = bib.get('pub_year', 'N/A')
                
                # 只有当包含有效摘要时才保存（用于分类）
                if abstract != 'N/A' and len(abstract) > 50:
                    fetched_data.append({
                        'search_query_cluster': cluster_id, # 记录这是用哪个领域的词搜出来的
                        'title': title,
                        'abstract': abstract,
                        'url': pub_url,
                        'year': year
                    })
                    print(f"  [Found] {title[:60]}...")
                    count += 1
                
                # 随机休眠 2-5 秒，防止被封 IP
                time.sleep(random.uniform(2, 5))
                
        except Exception as e:
            print(f"  [Error] searching for cluster {cluster_id}: {e}")
            continue
            
    return pd.DataFrame(fetched_data)

# ==========================================
# 4. 主执行流程
# ==========================================

def main():
    # --- 步骤 1: 爬取数据 ---
    # 为了演示，这里每个领域只爬取 3 篇。实际使用可以改大 max_per_query
    print("--- Step 1: Fetching papers from Google Scholar ---")
    new_papers_df = fetch_papers_from_google_scholar(SEARCH_QUERIES, max_per_query=3)
    
    if new_papers_df.empty:
        print("No papers found or crawler blocked. Exiting.")
        return

    print(f"\nFetched {len(new_papers_df)} papers. Now classifying...")

    # --- 步骤 2: 初始化分类器 ---
    # 确保当前目录下有 .csv 和 .npy 文件
    if not os.path.exists(EXISTING_CSV_PATH) or not os.path.exists(EXISTING_EMBEDDINGS_PATH):
        print("Error: content files (csv/npy) not found.")
        return
        
    classifier = PaperClassifier(EXISTING_CSV_PATH, EXISTING_EMBEDDINGS_PATH, MODEL_NAME)

    # --- 步骤 3: 分类与结果处理 ---
    results = []
    
    for idx, row in new_papers_df.iterrows():
        title = row['title']
        abstract = row['abstract']
        
        # 调用分类
        pred_cluster, score, _ = classifier.classify(title, abstract)
        
        # 结果格式化
        status = "Unrelated" if pred_cluster == -1 else f"Cluster {pred_cluster}"
        cluster_name = "Unrelated / Out of Scope" if pred_cluster == -1 else CLUSTER_NAMES[pred_cluster]
        
        results.append({
            'title': title,
            'abstract': abstract,
            'url': row['url'],
            'year': row['year'],
            'search_origin': row['search_query_cluster'], # 它是用哪个词搜出来的
            'predicted_cluster': pred_cluster,            # 模型判定它属于哪个类
            'predicted_label': cluster_name,
            'confidence_score': round(score, 4)
        })
        
        print(f"Paper: {title[:30]}... -> {status} (Score: {score:.2f})")

    # --- 步骤 4: 保存结果 ---
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\nDone! Results saved to {OUTPUT_CSV_PATH}")
    print(final_df[['title', 'predicted_label', 'confidence_score']].head())

if __name__ == "__main__":
    main()