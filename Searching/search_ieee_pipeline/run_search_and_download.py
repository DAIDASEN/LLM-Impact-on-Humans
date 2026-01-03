$env:PYTHONPATH=$PWD.Path; conda activate Embedding; python search_ieee_pipeline/search_arxiv.py --max 50 --download#!/usr/bin/env python3
"""Integrate `Search.py` queries with `IEEE_downloader` to fetch PDFs per cluster.

Place this file in the workspace root under `search_ieee_pipeline/`.
Usage: import functions from this module or run as script.
"""
import time
import random
import os
import sys
# ensure workspace root is on sys.path so sibling modules (e.g. Search.py) import cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Search import BROAD_SEARCH_QUERIES, CLUSTER_NAMES

# Import IEEE_downloader functions (keep original package untouched)
import importlib

# Ensure IEEE_downloader package modules import correctly even though some modules
# use top-level `from utils import ...`. We load the package utils module and
# register it as top-level 'utils' so those imports resolve.
pkg_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "IEEE_downloader")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

ieee_pkg_utils = importlib.import_module("IEEE_downloader.utils")
sys.modules.setdefault("utils", ieee_pkg_utils)

from IEEE_downloader.download_paper_by_pageURL import organize_info_by_query
from IEEE_downloader import utils as ieee_utils


def run_cluster_download(cluster_id, pages, save_root="./ieee_pipeline_outputs", add_year=True, sleep_between_clusters=(2,4)):
    """Run download for a single cluster using the query from `Search.py`.

    Args:
        cluster_id (int): cluster id present in `BROAD_SEARCH_QUERIES`.
        pages (list[int]): list of page numbers to fetch from IEEE search API.
        save_root (str): root folder to save PDFs per-cluster.
        add_year (bool): whether to prefix saved filenames with year.
        sleep_between_clusters (tuple): random sleep range after finishing cluster.

    Returns:
        dict: summary with keys: status (bool), downloaded_count (int), already_exist (int)
    """
    if cluster_id not in BROAD_SEARCH_QUERIES:
        raise KeyError(f"cluster_id {cluster_id} not in BROAD_SEARCH_QUERIES")

    query = BROAD_SEARCH_QUERIES[cluster_id]
    save_dir = os.path.join(save_root, f"cluster_{cluster_id}")
    os.makedirs(save_dir, exist_ok=True)

    # initialize progress store used by IEEE_downloader utils
    ieee_utils._init()

    print(f"[Pipeline] Cluster {cluster_id}: {CLUSTER_NAMES.get(cluster_id)}")
    print(f"[Pipeline] Query: {query}")
    print(f"[Pipeline] Pages: {pages}")
    try:
        status, paper_info = organize_info_by_query(query, pages, save_dir, paper_name_with_year=add_year)
    except Exception as e:
        print(f"[Error] organize_info_by_query failed: {e}")
        return {"status": False, "downloaded": 0, "already_exist": 0}

    if not status or not paper_info:
        print("[Pipeline] No papers parsed for this query.")
        return {"status": False, "downloaded": 0, "already_exist": 0}

    succeed, downloaded, already = ieee_utils.downLoad_paper(paper_info, show_bar=False)
    # gentle pause to reduce risk of rate limit when looping clusters
    time.sleep(random.uniform(*sleep_between_clusters))
    return {"status": succeed, "downloaded": downloaded, "already_exist": already}


def run_multiple_clusters(cluster_ids, pages_per_cluster, save_root="./ieee_pipeline_outputs", add_year=True):
    results = {}
    for cid in cluster_ids:
        res = run_cluster_download(cid, pages_per_cluster, save_root=save_root, add_year=add_year)
        results[cid] = res
    return results


if __name__ == "__main__":
    # Example usage: download first page for all defined BROAD_SEARCH_QUERIES
    cluster_ids = list(BROAD_SEARCH_QUERIES.keys())
    # request ~100 papers per cluster -> IEEE returns up to 25 per page, request pages 1..4
    pages = [1,2,3,4]
    print("Starting pipeline: requesting pages 1-4 (~100 items per cluster)")
    out = run_multiple_clusters(cluster_ids, pages, save_root="./ieee_pipeline_outputs", add_year=True)
    print("Finished. Summary:")
    for cid, v in out.items():
        print(f" cluster {cid}: {v}")
