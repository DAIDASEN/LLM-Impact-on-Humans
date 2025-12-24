import time
import random
import requests
import pandas as pd
import numpy as np
from scholarly import scholarly
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ==========================================
# 1. 核心配置与文件路径
# ==========================================

EXISTING_CSV_PATH = 'papers_with_clusters.csv'
EXISTING_EMBEDDINGS_PATH = 'embeddings.npy'
OUTPUT_CSV_PATH = 'enhanced_experimental_papers.csv'
MODEL_NAME = 'all-mpnet-base-v2' # 需与你生成 npy 时用的模型一致
SIMILARITY_THRESHOLD = 0.35      # 低于此相似度则视为无关

# ==========================================
# 2. 增强版搜索指令 (Enhanced Search Queries)
# ==========================================
# 这些 Query 是根据你提供的详细列表构建的，专为 Google Scholar 设计。
# 逻辑：(核心技术) AND (丰富领域词) AND (实验验证) -排除词
#TODO : 添加Human的词汇
SEARCH_QUERIES = {
    # Cluster 0: Social & Collaboration
    0: """("LLM" OR "large language model" OR "generative AI" OR "ChatGPT" OR "AI chatbot" OR "LLMs" OR "large language models")
       AND ("AI companionship" OR "AI companion" OR "human-AI relationship" OR "emotional support" OR "loneliness" OR "anthropomorphism" OR "parasocial relationship" OR "mental health" OR "user-driven value alignment") 
       AND ("user study" OR "randomized controlled trial" OR "field experiment" OR "participants" OR "N=" OR "empirical study") 
       -survey -review -roadmap -theoretical""",

    # Cluster 1: Psychology & Persuasion
    1: """("LLM" OR "LLMs" OR "large language model" OR "large language models"
            OR "language model" OR "language models"
            OR "generative AI" OR "ChatGPT" OR "AI assistant" OR "conversational AI" OR "AI chatbot")
       AND ("conspiracy belief" OR "conspiracy beliefs" OR "conspiracy theory" OR "conspiracy theories"
            OR misinformation OR disinformation OR "false information" OR "fact-checking" OR "fact checking"
            OR "headline discernment" OR "truth discernment"
            OR persuasion OR "political persuasion" OR "persuasive message" OR "persuasive messages"
            OR "attitude change" OR "opinion change" OR "belief change" OR "belief updating" OR "belief revision"
            OR "issue framing" OR reframing OR "cognitive reframing" OR "cognitive reappraisal"
            OR "policy issue" OR "policy issues"
            OR election OR elections OR voters OR voting
            OR ideology OR ideological OR partisanship OR partisan OR "political leaning" OR "political bias"
            OR polarization OR polarisation OR depolarization OR depolarisation
            OR deliberation OR "democratic deliberation" OR "deliberative democracy" OR "deliberative polling"
            OR "common ground" OR consensus OR "collective intelligence"
            OR "content moderation")
       AND (trust OR reliance OR overreliance OR "user reliance"
            OR "trust calibration" OR calibration OR miscalibration
            OR "algorithmic aversion" OR "algorithm aversion" OR "algorithmic appreciation"
            OR "automation bias" OR "advice taking" OR "advice-taking"
            OR "decision support" OR "human-AI decision-making"
            OR "uncertainty expression" OR "uncertainty communication" OR hedging OR confidence OR overconfidence)
       AND ("user study" OR users OR "human subjects"
            OR randomized OR randomised OR "randomized controlled trial" OR "randomised controlled trial"
            OR preregistered OR "pre-registered"
            OR "controlled experiment" OR "lab experiment" OR "online experiment" OR "field experiment"
            OR participants OR "N=" OR "empirical study" OR "behavioral experiment")
       -survey -questionnaire -review -meta-analysis -systematic -roadmap -theoretical -conceptual -framework""",

    # Cluster 2: Creativity & Ideation
    2: """("LLM" OR "large language model" OR "generative AI" OR "ChatGPT" OR "LLMs" OR "large language models") 
       AND ("creativity" OR "creative writing" OR "brainstorming" OR "divergent thinking" OR "collective intelligence" OR "teamwork" OR "collaborating with AI agents" OR "novelty" OR "writing style homogenization") 
       AND ("user study" OR "randomized controlled trial" OR "field experiment" OR "participants" OR "N=" OR "empirical study") 
       -survey -review -roadmap -theoretical""",

    # Cluster 3: Education & Productivity
    3: """("LLM" OR "LLMs" OR "large language model" OR "large language models"
            OR "generative AI" OR "generative artificial intelligence"
            OR "ChatGPT" OR "AI assistant" OR "AI tutor"
            OR Copilot OR "GitHub Copilot" OR "AI writing assistant" OR "conversational agent")
       AND (education OR learning OR classroom OR student OR students OR "higher education" OR university OR college
            OR school OR "high school" OR "secondary school"
            OR productivity OR "knowledge work" OR workplace OR "task performance"
            OR "professional writing" OR "writing task" OR writing OR "essay writing"
            OR "writing assistant" OR "writing assistance" OR "writing skills"
            OR feedback OR tutoring OR "homework tutor"
            OR "learning outcomes" OR "learning performance" OR achievement OR grades
            OR engagement OR motivation OR persistence
            OR metacognition OR metacognitive OR "self-regulated learning"
            OR "metacognitive laziness" OR "cognitive debt"
            OR "cognitive load" OR "critical thinking"
            OR "reading comprehension" OR "note-taking" OR "memory retention"
            OR programming OR "programming assignment" OR "code completion"
            OR "automated assessment" OR "compiler error"
            OR mathematics OR math OR "high school math"
            OR "EFL writing"
            OR "assistive technology" OR "visual disability" OR "visual impairment")
       AND ("user study" OR users OR "human subjects"
            OR randomized OR randomised OR "randomized controlled trial" OR "randomised controlled trial"
            OR preregistered OR "pre-registered"
            OR "controlled experiment" OR "lab experiment" OR "online experiment" OR "field experiment"
            OR participants OR students OR "N=" OR "empirical study")
       -survey -questionnaire -review -meta-analysis -systematic -roadmap -theoretical -conceptual -framework"""
}

CLUSTER_NAMES = {
    0: "Cluster 0: Social & Collaboration",
    1: "Cluster 1: Psychology & Persuasion",
    2: "Cluster 2: Creativity & Ideation",
    3: "Cluster 3: Education & Productivity"
}

# ==========================================
# 3. 辅助函数：Semantic Scholar API
# ==========================================
def get_full_abstract_from_s2(title):
    """
    通过标题去 Semantic Scholar 获取完整摘要
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "limit": 1,
            "fields": "title,abstract,year,url,venue"
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data['total'] > 0 and 'data' in data:
                paper = data['data'][0]
                return paper.get('abstract'), paper.get('url')
    except Exception as e:
        pass # 静默失败，使用 Google Snippet 作为备选
    
    return None, None

# ==========================================
# 4. 核心类：分类器
# ==========================================
class PaperClassifier:
    def __init__(self, csv_path, npy_path, model_name):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("Loading existing data and calculating centroids...")
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(npy_path)
        self.centroids = self._calculate_centroids()

    def _calculate_centroids(self):
        centroids = {}
        for cluster_id in range(4):
            indices = self.df[self.df['cluster'] == cluster_id].index
            if len(indices) > 0:
                cluster_embeddings = self.embeddings[indices]
                centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
            else:
                centroids[cluster_id] = np.zeros(768)
        return centroids

    def classify(self, title, abstract):
        text = f"{title} [SEP] {abstract}"
        vector = self.model.encode([text])
        
        scores = {}
        for cid, centroid in self.centroids.items():
            sim = cosine_similarity(vector, centroid.reshape(1, -1))[0][0]
            scores[cid] = sim
            
        best_cluster = max(scores, key=scores.get)
        best_score = scores[best_cluster]
        
        if best_score < SIMILARITY_THRESHOLD:
            return -1, best_score, scores
        else:
            return best_cluster, best_score, scores

# ==========================================
# 5. 主程序逻辑
# ==========================================
def main():
    # --- 步骤 1: 爬取与初步过滤 ---
    fetched_data = []
    # 实验性验证词 (在摘要中查找)
    experimental_keywords = ["participant", "user study", "experiment", "n=", "subjects", "respondents", "empirical", "rct", "method", "interview", "survey"]
    
    print("Starting search with enhanced queries...")
    
    for cluster_id, query in SEARCH_QUERIES.items():
        print(f"\n--- Searching Cluster {cluster_id} ---")
        try:
            # 注意: 这里每组只取 10 篇演示，实际使用可调大
            search_iterator = scholarly.search_pubs(query)
            count = 0
            
            for pub in search_iterator:
                if count >= 10: break # 每个 Cluster 限制抓取数量
                
                title = pub.get('bib', {}).get('title')
                if not title: continue
                
                print(f"  [Found] {title[:50]}...")
                
                # 双层抓取：去 S2 拿完整摘要
                full_abstract, s2_url = get_full_abstract_from_s2(title)
                
                final_abstract = full_abstract if full_abstract else pub.get('bib', {}).get('abstract', '')
                final_url = s2_url if s2_url else pub.get('pub_url')
                
                # 过滤逻辑：必须包含实验性词汇
                if final_abstract and len(final_abstract) > 50:
                    is_experimental = any(k in final_abstract.lower() for k in experimental_keywords)
                    
                    if is_experimental:
                        fetched_data.append({
                            'search_cluster': cluster_id,
                            'title': title,
                            'abstract': final_abstract,
                            'url': final_url,
                            'year': pub.get('bib', {}).get('pub_year', 'N/A')
                        })
                        print(f"    -> [Kept] Verified as experimental.")
                        count += 1
                    else:
                        print(f"    -> [Dropped] Likely theoretical/review.")
                
                # 礼貌休眠
                time.sleep(random.uniform(2, 4))
                
        except Exception as e:
            print(f"  [Error] {e}")

    # --- 步骤 2: 分类与保存 ---
    if not fetched_data:
        print("No papers found matching criteria.")
        return

    print(f"\nClassifying {len(fetched_data)} papers...")
    
    if os.path.exists(EXISTING_CSV_PATH) and os.path.exists(EXISTING_EMBEDDINGS_PATH):
        classifier = PaperClassifier(EXISTING_CSV_PATH, EXISTING_EMBEDDINGS_PATH, MODEL_NAME)
        
        results = []
        for item in fetched_data:
            c_id, score, _ = classifier.classify(item['title'], item['abstract'])
            
            label = CLUSTER_NAMES.get(c_id, "Unrelated") if c_id != -1 else "Unrelated"
            
            results.append({
                **item,
                'predicted_cluster': c_id,
                'predicted_label': label,
                'confidence': round(score, 4)
            })
            print(f"  -> {label} (Score: {score:.2f}) | {item['title'][:30]}...")
            
        # 保存结果
        pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\nDone! Saved to {OUTPUT_CSV_PATH}")
    else:
        print("Error: Missing 'papers_with_clusters.csv' or 'embeddings.npy'. Cannot classify.")

if __name__ == "__main__":
    main()