# Search -> IEEE 下载集成说明

简要说明如何使用 `Search.py` 中的查询（`BROAD_SEARCH_QUERIES`）驱动 `IEEE_downloader` 批量下载论文。

准备
- 确保已安装依赖（工作区已包含 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

（`requirements.txt` 已包含 `scholarly`、`requests`、`pandas` 等，若缺少请手动安装）

使用步骤
1. 在工作区根目录打开终端。
2. 运行示例脚本（一次为每个 cluster 下载第 1 页）：

```bash
python search_ieee_pipeline/run_search_and_download.py
```

3. 定制下载：在脚本中或从 Python 导入 `run_cluster_download` / `run_multiple_clusters`：

```python
from search_ieee_pipeline.run_search_and_download import run_cluster_download, run_multiple_clusters

# 单 cluster 示例：下载 cluster 0 的第 1~2 页
run_cluster_download(cluster_id=0, pages=[1,2], save_root='./my_ieee_save', add_year=True)

# 多 cluster 批量运行
run_multiple_clusters([0,1], pages_per_cluster=[1], save_root='./my_ieee_save')
```

注意事项
- 本集成把下载输出写到 `./ieee_pipeline_outputs/cluster_{cid}` 下，确保有写权限。
- `organize_info_by_query` 使用 IEEE 的 REST 搜索接口，实际返回受查询语法和网络权限影响，建议先做小规模（单页）测试并根据返回调整查询字符串。
- 请在允许下载 IEEE 资源的网络环境下运行（学校/机构 VPN 或 IP 白名单）。
- 若大量抓取，请适当增加等待时间以降低被限流风险。

如果你希望我把脚本放到不同位置、添加命令行参数解析（`argparse`）或把下载结果写入日志/CSV，请告诉我，我可以继续完善。 
