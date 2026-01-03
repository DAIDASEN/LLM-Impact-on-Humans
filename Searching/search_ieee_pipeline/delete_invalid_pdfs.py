import pathlib, csv

ROOTS = [pathlib.Path('ieee_pipeline_outputs'), pathlib.Path('arxiv_pipeline_outputs'), pathlib.Path('papers'), pathlib.Path('papers_bad')]
MIN_SIZE = 2000
out_log = pathlib.Path('search_ieee_pipeline/deleted_invalid_files.csv')
out_log.parent.mkdir(exist_ok=True)

deleted = []
for root in ROOTS:
    if not root.exists():
        continue
    for p in root.rglob('*.pdf'):
        try:
            ok = p.stat().st_size >= MIN_SIZE and p.read_bytes()[:4] == b'%PDF'
        except Exception:
            ok = False
        if not ok:
            try:
                deleted.append({'path': str(p), 'size': p.stat().st_size if p.exists() else 0})
                p.unlink()
            except Exception as e:
                deleted.append({'path': str(p), 'size': 'unlink_failed'})

with open(out_log, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['path','size'])
    w.writeheader()
    for d in deleted:
        w.writerow(d)

print(f"Deleted {len(deleted)} invalid files. Log: {out_log}")
