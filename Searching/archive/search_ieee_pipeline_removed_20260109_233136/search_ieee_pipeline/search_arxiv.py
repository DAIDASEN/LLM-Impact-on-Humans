import os
import time
import csv
import requests
import urllib.parse
import xml.etree.ElementTree as ET

from Search import BROAD_SEARCH_QUERIES

ARXIV_API = "http://export.arxiv.org/api/query"


def query_arxiv(query, start=0, max_results=50, timeout=20):
    q = urllib.parse.quote_plus(query)
    url = f"{ARXIV_API}?search_query=all:{q}&start={start}&max_results={max_results}"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "arXivSearch/1.0"})
    resp.raise_for_status()
    return resp.text


def parse_arxiv_feed(feed_xml):
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(feed_xml)
    entries = []
    for entry in root.findall("atom:entry", ns):
        e = {}
        id_text = entry.find("atom:id", ns)
        if id_text is None:
            continue
        # id like http://arxiv.org/abs/2101.00001v1
        arxiv_id = id_text.text.strip().split('/')[-1]
        e['arxiv_id'] = arxiv_id
        title = entry.find("atom:title", ns)
        e['title'] = title.text.strip().replace('\n', ' ') if title is not None else ''
        summary = entry.find("atom:summary", ns)
        e['summary'] = summary.text.strip().replace('\n', ' ') if summary is not None else ''
        published = entry.find("atom:published", ns)
        e['published'] = published.text if published is not None else ''
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None]
        e['authors'] = '; '.join(authors)
        e['pdf_url'] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        entries.append(e)
    return entries


def save_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = ['title', 'arxiv_id', 'pdf_url', 'summary', 'published', 'authors', 'source']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def download_pdf(url, out_path, timeout=30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "arXivDownloader/1.0"})
        resp.raise_for_status()
        # basic content-type check
        ct = resp.headers.get('Content-Type', '')
        if 'pdf' not in ct.lower():
            return False
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def run(max_per_query=50, pause_between=3, download=False, out_dir='arxiv_pipeline_outputs'):
    all_rows = []
    for cid, q in BROAD_SEARCH_QUERIES.items():
        print(f"Querying arXiv for cluster {cid}: '{q}'")
        try:
            feed = query_arxiv(q, max_results=max_per_query)
        except Exception as e:
            print(f"Failed query '{q}': {e}")
            continue
        entries = parse_arxiv_feed(feed)
        for e in entries:
            row = {
                'title': e['title'],
                'arxiv_id': e['arxiv_id'],
                'pdf_url': e['pdf_url'],
                'summary': e['summary'],
                'published': e['published'],
                'authors': e['authors'],
                'source': 'arxiv'
            }
            all_rows.append(row)
            if download:
                pdf_dir = os.path.join(out_dir, f"cluster_{cid}")
                pdf_path = os.path.join(pdf_dir, f"{e['arxiv_id']}.pdf")
                ok = download_pdf(e['pdf_url'], pdf_path)
                if not ok:
                    print(f"Download failed for {e['arxiv_id']}")
        time.sleep(pause_between)

    csv_out = os.path.join('search_ieee_pipeline', 'sources', 'arxiv', 'papers_for_pipeline_arxiv.csv')
    # annotate rows with source
    for r in all_rows:
        r['source'] = 'arxiv'
    save_csv(all_rows, csv_out)
    print(f"Wrote {len(all_rows)} records to {csv_out}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Search arXiv using BROAD_SEARCH_QUERIES and export CSV')
    p.add_argument('--max', type=int, default=50, help='max results per query')
    p.add_argument('--download', action='store_true', help='also download PDFs')
    p.add_argument('--pause', type=float, default=3.0, help='seconds between queries')
    args = p.parse_args()
    run(max_per_query=args.max, pause_between=args.pause, download=args.download)
