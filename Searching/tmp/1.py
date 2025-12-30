#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → PDF 批量下载工具（适配 stage1_titles.csv）
用法:
  python download_pdfs.py stage1_titles.csv out_pdfs --width 4 --sleep 1
"""

import argparse, csv, io, os, re, sys, time, zipfile
from urllib.parse import urlparse, parse_qs, quote
import requests

PDF_MAGIC = b"%PDF-"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

ARXIV_ABS_RE = re.compile(r"^https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(v\d+)?/?$")
ARXIV_PDF_RE = re.compile(r"^https?://arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})(v\d+)?(\.pdf)?/?$")
ACL_RE = re.compile(r"^https?://aclanthology\.org/([^/]+)/?$", re.I)
OPENREVIEW_FORUM_RE = re.compile(r"^https?://openreview\.net/forum", re.I)
OPENREVIEW_PDF_RE = re.compile(r"^https?://openreview\.net/pdf", re.I)

def zero_pad(n: int, width: int) -> str:
    return str(n).zfill(width)

def normalize_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    # arXiv abs → pdf
    m = ARXIV_ABS_RE.match(u)
    if m:
        arxiv_id, v = m.group(1), m.group(2) or ""
        return f"https://arxiv.org/pdf/{arxiv_id}{v}.pdf"
    # arXiv pdf 补全
    m = ARXIV_PDF_RE.match(u)
    if m:
        arxiv_id, v, ext = m.group(1), m.group(2) or "", m.group(3) or ""
        return f"https://arxiv.org/pdf/{arxiv_id}{v}.pdf" if ext.lower() != ".pdf" else u
    # ACL
    if ACL_RE.match(u) and not u.lower().endswith(".pdf"):
        return u.rstrip("/") + ".pdf"
    # OpenReview forum → pdf
    if OPENREVIEW_FORUM_RE.match(u) and not OPENREVIEW_PDF_RE.match(u):
        pid = (parse_qs(urlparse(u).query).get("id") or [""])[0]
        if pid:
            return f"https://openreview.net/pdf?id={quote(pid)}"
    return u

def is_probably_pdf(resp: requests.Response, first_bytes: bytes) -> bool:
    return "application/pdf" in (resp.headers.get("Content-Type") or "").lower() or first_bytes.startswith(PDF_MAGIC)

def download_pdf(url: str, out_path: str, timeout: int = 10, max_retries: int = 3) -> (bool, str):
    headers = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"}
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(1.0 * attempt)
                    continue
                first = next(r.iter_content(chunk_size=8192), b"")
                if not is_probably_pdf(r, first):
                    last_err = f"Not PDF (CT={r.headers.get('Content-Type')})"
                    time.sleep(0.8 * attempt)
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(first)
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True, "ok"
        except Exception as e:
            last_err = repr(e)
            time.sleep(1.0 * attempt)
    return False, last_err or "failed"

def read_rows_from_input(input_path: str):
    if input_path.lower().endswith(".zip"):
        with zipfile.ZipFile(input_path, "r") as z:
            names = [n for n in z.namelist() if n.lower().endswith((".csv", ".tsv"))]
            if not names:
                raise RuntimeError("ZIP 里没有 CSV/TSV")
            name = names[0]
            raw = z.read(name).decode("utf-8", errors="replace")
            f, delim = io.StringIO(raw), ("\t" if name.lower().endswith(".tsv") else ",")
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                yield row
    else:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            delim = "\t" if input_path.lower().endswith(".tsv") else ","
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                yield row

def pick_index_and_url(row: dict, line_no: int):
    # 去BOM/空格
    row = {k.strip("\ufeff \t\r\n"): v for k, v in row.items()}
    url = row.get("url_gs", "").strip()
    # 用行号当 index
    return line_no, url

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV/TSV 或 ZIP")
    ap.add_argument("outdir", help="输出目录")
    ap.add_argument("--width", type=int, default=3, help="零填充宽度")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=0, help="只跑前 N 行（0=全部）")
    args = ap.parse_args()

    input_path, outdir = args.input, args.outdir
    os.makedirs(outdir, exist_ok=True)

    ok_cnt = fail_cnt = 0
    fail_log = os.path.join(outdir, "failures.tsv")
    with open(fail_log, "w", encoding="utf-8") as flog:
        flog.write("index\turl\treason\tfallback_url\n")
        for i, row in enumerate(read_rows_from_input(input_path), start=1):
            if args.max and i > args.max:
                break
            idx, url = pick_index_and_url(row, i)
            if not url:
                continue
            url_norm = normalize_url(url)
            filename = f"{zero_pad(idx, args.width)}.pdf"
            out_path = os.path.join(outdir, filename)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                print(f"[SKIP] {filename} 已存在")
                ok_cnt += 1
                continue

            ok, reason = download_pdf(url_norm, out_path, timeout=args.timeout, max_retries=args.retries)
            if ok:
                ok_cnt += 1
                print(f"[OK ] {filename}")
            else:
                fail_cnt += 1
                print(f"[FAIL] {filename} 原因: {reason}")
                flog.write(f"{idx}\t{url}\t{reason}\t\n")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"\n完成！成功 {ok_cnt} 篇，失败 {fail_cnt} 篇，日志见 {fail_log}")

if __name__ == "__main__":
    main()