## 三段 Prompt 对应的 3 个脚本（PDF 批处理）

你设计的流程是：
- **Stage 1**（`validate_paper.py`）：判断 `PDF_TEXT` 是否像论文，并校验题目/作者与 `SEARCH_RECORD` 是否匹配。
- **Stage 2**（`screen_cluster.py`）：严格实证筛选（是否“LLM 对人影响”的实证研究），并分到簇（0-3/4/-1）。
- **Stage 3**（`extract_summary.py`）：只对满足门槛的论文抽取“LLM 对人影响”的结构化信息（见下方门槛）。

推荐使用 **方案A（不复制 PDF，不创建 `papers_curated/` 目录）**：
- Stage 1 通过后，只生成一个“子集 CSV”（`llm_outputs/papers_curated.csv`）
- Stage 2/3 继续从原始 `papers/` 读取 PDF
- 好处：**不重复占用磁盘空间**

### 0) 安装依赖

```bash
pip install -r requirements.txt
```

### 1) 配置 API Key（不泄露）

推荐用环境变量（Windows PowerShell）：

```powershell
$env:OPENAI_API_KEY="你的key"
```

可选：在项目根目录创建 `.env`（已在 `.gitignore` 中忽略，不会提交）：

```ini
OPENAI_API_KEY=你的key
OPENAI_MODEL=gpt-5
```

### 2) 运行 Stage 1：论文 + 匹配校验（可断点续跑）

```bash
python validate_paper.py --resume --csv papers.csv --papers_dir papers --out llm_outputs/validate.jsonl --inplace
```

### 3) 生成子集 CSV（方案A：不复制PDF）

```bash
# 默认不会复制/移动 PDF，只会生成 llm_outputs/papers_curated.csv（文件名沿用，实际不需要 papers_curated/ 目录）
python run_pipeline.py --curate_mode none
```

生成的子集 CSV：
- `llm_outputs/papers_curated.csv`（仅包含 Stage1 通过：is_paper==True 且 is_match==True 的论文）

### 4) 运行 Stage 2：实证筛选 + 聚类（只跑子集，且仍从 papers/ 读 PDF）

```bash
python screen_cluster.py --resume --csv llm_outputs/papers_curated.csv --papers_dir papers --out llm_outputs/screen_cluster.jsonl
```

### 5) 运行 Stage 3：信息抽取（依赖 Stage 1 + Stage 2 输出）

Stage 3 门槛（必须同时满足）：
- Stage 1：`is_paper == True` 且 `is_match == True`
- Stage 2：`cluster_id != -1`（不为 Excluded）

```bash
python extract_summary.py --resume --csv llm_outputs/papers_curated.csv --papers_dir papers --stage1_jsonl llm_outputs/validate.jsonl --screen_jsonl llm_outputs/screen_cluster.jsonl --out llm_outputs/extract_summary.jsonl --only_included --require_step1_pass
```

### 6) 输出说明（最后看哪个）

所有输出都是 `jsonl`（一行一个 JSON），方便你后续用 pandas 读回去再转 CSV。

你通常最终看：
- `llm_outputs/extract_summary.jsonl`（Stage 3 的结构化抽取结果）

读回示例：

```python
import pandas as pd
df = pd.read_json("llm_outputs/extract_summary.jsonl", lines=True)
```



