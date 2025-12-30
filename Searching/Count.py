import re
from pathlib import Path
import pandas as pd

CSV_PATH = "stage1_titles.csv"   # 你的原始CSV（带 cluster 列）
PAPERS_DIR = Path("papers")      # 你手动下载的PDF所在目录

# 1) 读CSV：df.index 就是“原始序号”
df = pd.read_csv(CSV_PATH)
df["row_id"] = df.index

# 2) 扫描 papers 目录里的 PDF，并提取“文件名前缀数字”作为 row_id
# 支持：0001.pdf / 0001_xxx.pdf / 0001-xxx.pdf
id_pat = re.compile(r"^(\d+)")
downloaded_ids = set()

if PAPERS_DIR.exists():
    for p in PAPERS_DIR.glob("*.pdf"):
        m = id_pat.match(p.stem)  # p.stem = 去掉 .pdf 的文件名
        if m:
            downloaded_ids.add(int(m.group(1)))
else:
    raise SystemExit(f"找不到目录: {PAPERS_DIR.resolve()}")

df["downloaded"] = df["row_id"].isin(downloaded_ids)

# 3) 按 cluster 汇总
summary = (
    df.groupby(["search_cluster", "search_cluster_name"], dropna=False)
      .agg(total=("title", "size"), downloaded=("downloaded", "sum"))
      .reset_index()
      .sort_values("search_cluster")
)

summary["coverage_pct"] = (summary["downloaded"] / summary["total"] * 100).round(1)

print(summary.to_string(index=False))

# 4) 保存结果
summary.to_csv("download_count_by_cluster.csv", index=False, encoding="utf-8")
df[["row_id", "search_cluster", "search_cluster_name", "title", "downloaded"]].to_csv(
    "download_status_by_row.csv", index=False, encoding="utf-8"
)

print("\n已输出：download_count_by_cluster.csv / download_status_by_row.csv")
print(f"识别到已下载PDF数量：{len(downloaded_ids)}（仅统计文件名前缀是数字的PDF）")
