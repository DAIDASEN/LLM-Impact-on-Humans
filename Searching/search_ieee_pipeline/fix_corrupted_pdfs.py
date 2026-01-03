#!/usr/bin/env python3
"""Scan `papers/` for corrupted PDFs and attempt to re-download using mapping CSVs.

Usage: python search_ieee_pipeline/fix_corrupted_pdfs.py
"""
import os
import pathlib
import csv
import requests
import shutil

PAPERS_DIR = pathlib.Path("papers")
MAPPING_GLOB = pathlib.Path("ieee_pipeline_outputs").rglob("download_mapping.csv")
MIN_SIZE = 2000  # bytes


def is_valid_pdf(path: pathlib.Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < MIN_SIZE:
            return False
        with open(path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


def load_mappings():
    mappings = {}
    for p in pathlib.Path("ieee_pipeline_outputs").rglob("download_mapping.csv"):
        try:
            with open(p, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    # expect [requested_name, saved_name, url]
                    if len(row) >= 3:
                        requested, saved, url = row[0], row[1], row[2]
                        mappings[pathlib.Path(saved).name] = url
                        mappings[pathlib.Path(requested).name] = url
        except Exception:
            continue
    return mappings


def try_redownload(url, dest: pathlib.Path, session: requests.Session) -> bool:
    try:
        r = session.get(url, timeout=30)
        content = r.content
        # if content looks like HTML error page, save as .html for inspection
        if content.strip().lower().startswith(b"<html") or not content.startswith(b"%PDF"):
            # save html for debugging
            dest_html = dest.with_suffix('.html')
            with open(dest_html, 'wb') as hf:
                hf.write(content)
            return False
        # write to temp then move
        tmp = dest.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            f.write(content)
        shutil.move(str(tmp), str(dest))
        return True
    except Exception as e:
        return False


def main():
    if not PAPERS_DIR.exists():
        print("papers/ directory not found")
        return
    mappings = load_mappings()
    corrupted = []
    for p in PAPERS_DIR.rglob('*.pdf'):
        if not is_valid_pdf(p):
            corrupted.append(p)

    print(f"Found {len(corrupted)} corrupted or invalid PDF files")
    if not corrupted:
        return

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    })

    fixed = 0
    failed = []
    for p in corrupted:
        name = p.name
        url = mappings.get(name)
        if not url:
            # try matching by prefix
            for k in mappings.keys():
                if name.startswith(k.split('.')[0]):
                    url = mappings[k]
                    break
        if not url:
            failed.append((p, 'no_mapping'))
            continue
        print(f"Re-downloading {p} from {url}")
        ok = try_redownload(url, p, session)
        if ok:
            fixed += 1
        else:
            failed.append((p, 'download_failed'))

    print(f"Fixed: {fixed}; Remaining failures: {len(failed)}")
    for p, reason in failed:
        print(p, reason)


if __name__ == '__main__':
    main()
