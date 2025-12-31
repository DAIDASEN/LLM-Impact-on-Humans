## 三段 Prompt 对应的 3 个脚本（PDF 批处理）

你设计的流程是：
- **Stage 1**（`validate_paper.py`）：判断 `PDF_TEXT` 是否像论文，并校验题目/作者与 `SEARCH_RECORD` 是否匹配。
- **Stage 2**（`screen_cluster.py`）：严格实证筛选（是否“LLM 对人影响”的实证研究），并分到簇（0-3/4/-1）。
- **Stage 3**（`extract_summary.py`）：在 Stage 2 的簇结果基础上，只抽取“LLM 对人影响”的结构化信息。

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

### 2) 运行 Stage 1：论文 + 匹配校验

```bash
python validate_paper.py --csv papers.csv --papers_dir papers --out llm_outputs/validate.jsonl
```

### 3) 运行 Stage 2：实证筛选 + 聚类

```bash
python screen_cluster.py --csv papers.csv --papers_dir papers --out llm_outputs/screen_cluster.jsonl
```

### 4) 运行 Stage 3：信息抽取（依赖 Stage 2 输出）

只对纳入（cluster_id != -1）的论文抽取：

```bash
python extract_summary.py --csv papers.csv --papers_dir papers --screen_jsonl llm_outputs/screen_cluster.jsonl --out llm_outputs/extract_summary.jsonl --only_included
```

### 5) 输出说明

所有输出都是 `jsonl`（一行一个 JSON），方便你后续用 pandas 读回去再转 CSV：

```python
import pandas as pd
df = pd.read_json("llm_outputs/extract_summary.jsonl", lines=True)
```



