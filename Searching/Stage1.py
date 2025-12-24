import time
import random
import re
import difflib
import pandas as pd
from scholarly import scholarly

# =========================
# 配置
# =========================
EXISTING_CSV_PATH = "papers_with_clusters.csv"   # 旧论文库（含 title, cluster）
STAGE1_TITLES_OUT = "stage1_titles.csv"          # 新搜索标题库
COVER_DETAIL_OUT = "stage1_coverage_detail.csv"  # 覆盖明细（逐条旧标题）
COVER_SUMMARY_OUT = "stage1_coverage_summary_by_cluster.csv"  # 每个cluster覆盖率汇总

# Stage 1：每个 cluster 抓多少条（越大越“广泛”，也越容易被GS限流）
STAGE1_LIMIT_PER_CLUSTER = 100

# 访问节流（降低被 Google Scholar 限制概率）
SLEEP_RANGE = (1.0, 2.0)

# 匹配策略：先 exact（归一化后的精确匹配），再 fuzzy 兜底
USE_FUZZY = True
FUZZY_THRESHOLD = 0.92  # 0.90~0.95 常用，越高越严格

# =========================
# Stage 1 Query：核心技术词 + 你预先设立的领域词（按 cluster）
# =========================
TECH_TERMS = '("LLM" OR "large language model" OR "generative AI" OR "ChatGPT" OR "AI chatbot")'

DOMAIN_TERMS = {
    0: '("AI companionship" OR "AI companion" OR "human-AI relationship" OR "emotional support" OR "loneliness" OR "anthropomorphism" OR "parasocial relationship" OR "mental health" OR "user-driven value alignment")',
    1: '("conspiracy beliefs" OR "political persuasion" OR "political bias" OR "moral decision" OR "misinformation" OR "trust in AI" OR "cognitive reframing" OR "opinion change" OR "decision-making")',
    2: '("creativity" OR "creative writing" OR "brainstorming" OR "divergent thinking" OR "collective intelligence" OR "teamwork" OR "collaborating with AI agents" OR "novelty" OR "writing style homogenization")',
    3: '("learning outcomes" OR "student engagement" OR "metacognition" OR "reading comprehension" OR "critical thinking" OR "programming education" OR "EFL writing" OR "writing skills" OR "AI in education")',
}

CLUSTER_NAMES = {
    0: "Cluster 0: Social & Collaboration",
    1: "Cluster 1: Psychology & Persuasion",
    2: "Cluster 2: Creativity & Ideation",
    3: "Cluster 3: Education & Productivity",
}

BROAD_SEARCH_QUERIES = {cid: f"{TECH_TERMS} AND {DOMAIN_TERMS[cid]}" for cid in DOMAIN_TERMS}


# =========================
# 工具：标题归一化（用于匹配）
# =========================
def normalize_title(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    t = re.sub(r"[\[\]\(\)\{\}\.,:;!?\"'“”‘’`´\-–—_/\\|<>~^+=*@#$%&]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================
# 读取旧库（title列固定为 title）
# =========================
def load_old_library(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "title" not in df.columns:
        raise ValueError(f"旧库 {path} 中未找到 title 列。实际列：{list(df.columns)}")
    if "cluster" not in df.columns:
        raise ValueError(f"旧库 {path} 中未找到 cluster 列（用于按cluster统计覆盖率）。实际列：{list(df.columns)}")

    out = df[["title", "cluster"]].copy()
    out["title"] = out["title"].astype(str)
    out["title_norm"] = out["title"].apply(normalize_title)

    # 去空与去重：以（cluster, title_norm）为粒度，避免同cluster重复标题干扰覆盖率
    out = out[out["title_norm"] != ""].drop_duplicates(["cluster", "title_norm"]).reset_index(drop=True)
    return out


# =========================
# Stage 1：广泛搜标题（只收集 title + 元信息）
# =========================
def stage1_search_titles(limit_per_cluster: int) -> pd.DataFrame:
    rows = []
    for cid, query in BROAD_SEARCH_QUERIES.items():
        print(f"\n[Stage 1] Searching: {CLUSTER_NAMES[cid]}")
        print(f"Query: {query}")

        try:
            it = scholarly.search_pubs(query)
            count = 0
            for pub in it:
                if count >= limit_per_cluster:
                    break
                if not isinstance(pub, dict):
                    continue

                bib = pub.get("bib", {}) or {}
                title = bib.get("title")
                if not title:
                    continue

                rows.append({
                    "search_cluster": cid,
                    "search_cluster_name": CLUSTER_NAMES.get(cid, str(cid)),
                    "title": title,
                    "title_norm": normalize_title(title),
                    "year_gs": bib.get("pub_year", ""),
                    "venue_gs": bib.get("venue", ""),
                    "author_gs": bib.get("author", ""),
                    "url_gs": pub.get("pub_url", ""),
                    "source": "google_scholar",
                })

                count += 1
                if count % 20 == 0:
                    print(f"  collected {count}/{limit_per_cluster} titles ...")

                time.sleep(random.uniform(*SLEEP_RANGE))

        except Exception as e:
            print(f"  [Stage 1 Error] cluster={cid}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(STAGE1_TITLES_OUT, index=False, encoding="utf-8-sig")
        return df

    # 全局去重（同一标题可能被多个query搜到）
    df = df[df["title_norm"] != ""].drop_duplicates(["title_norm"]).reset_index(drop=True)
    df.to_csv(STAGE1_TITLES_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[Stage 1] Saved {len(df)} unique new titles to {STAGE1_TITLES_OUT}")
    return df


# =========================
# Cover：计算旧标题在新标题中的覆盖情况，并按旧cluster统计覆盖率
# =========================
def compute_coverage(old_df: pd.DataFrame, new_df: pd.DataFrame):
    new_norm_to_title = dict(zip(new_df["title_norm"], new_df["title"]))
    new_norm_set = set(new_norm_to_title.keys())
    new_norm_list = list(new_norm_set)

    detail_rows = []

    for _, r in old_df.iterrows():
        old_title = r["title"]
        old_cluster = int(r["cluster"])
        old_norm = r["title_norm"]

        # 1) exact match
        if old_norm in new_norm_set:
            detail_rows.append({
                "old_cluster": old_cluster,
                "old_title": old_title,
                "match_type": "exact",
                "matched_new_title": new_norm_to_title.get(old_norm, ""),
                "match_score": 1.0,
            })
            continue

        # 2) fuzzy match (optional)
        if USE_FUZZY and old_norm:
            best = difflib.get_close_matches(old_norm, new_norm_list, n=1, cutoff=FUZZY_THRESHOLD)
            if best:
                matched_norm = best[0]
                score = difflib.SequenceMatcher(None, old_norm, matched_norm).ratio()
                detail_rows.append({
                    "old_cluster": old_cluster,
                    "old_title": old_title,
                    "match_type": f"fuzzy>={FUZZY_THRESHOLD}",
                    "matched_new_title": new_norm_to_title.get(matched_norm, ""),
                    "match_score": round(score, 4),
                })
                continue

        detail_rows.append({
            "old_cluster": old_cluster,
            "old_title": old_title,
            "match_type": "no_match",
            "matched_new_title": "",
            "match_score": "",
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(COVER_DETAIL_OUT, index=False, encoding="utf-8-sig")

    # 覆盖率汇总（按旧cluster）
    def hits(s): return (s != "no_match").sum()
    def exact(s): return (s == "exact").sum()

    summary_df = (
        detail_df.groupby("old_cluster")
        .agg(
            old_titles=("old_title", "count"),
            hits=("match_type", hits),
            exact=("match_type", exact),
        )
        .reset_index()
    )
    summary_df["fuzzy_hits"] = summary_df["hits"] - summary_df["exact"]
    summary_df["hit_rate"] = (summary_df["hits"] / summary_df["old_titles"]).round(4)
    summary_df["exact_rate"] = (summary_df["exact"] / summary_df["old_titles"]).round(4)

    summary_df.to_csv(COVER_SUMMARY_OUT, index=False, encoding="utf-8-sig")

    # 控制台输出：每个cluster覆盖率 + 覆盖了哪些（只打印前若干）
    print("\n[Coverage Summary by Old Cluster]")
    for _, row in summary_df.sort_values("old_cluster").iterrows():
        c = int(row["old_cluster"])
        print(
            f"  cluster={c} ({CLUSTER_NAMES.get(c,'')}): "
            f"old={int(row['old_titles'])}, hits={int(row['hits'])} ({row['hit_rate']:.2%}), "
            f"exact={int(row['exact'])} ({row['exact_rate']:.2%}), fuzzy={int(row['fuzzy_hits'])}"
        )

        covered_titles = detail_df[(detail_df["old_cluster"] == c) & (detail_df["match_type"] != "no_match")][
            ["old_title", "match_type", "matched_new_title", "match_score"]
        ]
        # 打印前 10 条示例，完整清单在 COVER_DETAIL_OUT
        if not covered_titles.empty:
            print("    covered examples (first 10):")
            for _, rr in covered_titles.head(10).iterrows():
                print(f"      - {rr['old_title']} | {rr['match_type']} | score={rr['match_score']}")

    print("\n[Outputs]")
    print(f"  Stage 1 titles: {STAGE1_TITLES_OUT}")
    print(f"  Coverage detail: {COVER_DETAIL_OUT}")
    print(f"  Coverage summary: {COVER_SUMMARY_OUT}")

    return detail_df, summary_df


def main():
    old_df = load_old_library(EXISTING_CSV_PATH)
    print(f"[Init] Loaded old library: {len(old_df)} unique titles (deduped per cluster)")

    new_df = stage1_search_titles(STAGE1_LIMIT_PER_CLUSTER)
    if new_df.empty:
        print("[Stop] Stage 1 returned empty results; cannot compute coverage.")
        return

    compute_coverage(old_df, new_df)


if __name__ == "__main__":
    main()
