import pathlib
import csv
import json
import re

OUT_DIR = pathlib.Path('search_ieee_pipeline')
OUT_DIR.mkdir(exist_ok=True)
ROOT = pathlib.Path('ieee_pipeline_outputs')

def is_pdf(p):
    try:
        return p.exists() and p.stat().st_size > 2000 and p.read_bytes()[:4] == b'%PDF'
    except Exception:
        return False


def load_mappings():
    # returns dict of basename -> url
    d = {}
    for csvp in ROOT.rglob('download_mapping.csv'):
        try:
            with open(csvp, encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                hdr = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        requested, saved, url = row[0].strip(), row[1].strip(), row[2].strip()
                        d[pathlib.Path(saved).name] = url
                        d[pathlib.Path(requested).name] = url
        except Exception:
            continue
    return d


def extract_arnumber(url: str):
    if not url:
        return ''
    m = re.search(r'arnumber=(\d+)', url)
    return m.group(1) if m else ''


def main():
    mappings = load_mappings()
    rows = []
    total = valid = invalid = 0
    for cluster_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        cluster_name = cluster_dir.name
        files = list(cluster_dir.glob('*.pdf'))
        for f in files:
            total += 1
            ok = is_pdf(f)
            if ok:
                valid += 1
            else:
                invalid += 1
            name = f.name
            url = mappings.get(name, '')
            arn = extract_arnumber(url)
            rows.append({
                'cluster': cluster_name,
                'filename': name,
                'path': str(f),
                'is_valid_pdf': ok,
                'url': url,
                'arnumber': arn,
                'size': f.stat().st_size if f.exists() else 0,
            })

    # write CSV and JSON
    csvp = OUT_DIR / 'download_report.csv'
    jsonp = OUT_DIR / 'download_report.json'
    with open(csvp, 'w', newline='', encoding='utf-8') as cf:
        w = csv.DictWriter(cf, fieldnames=['cluster','filename','path','is_valid_pdf','size','arnumber','url'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(jsonp, 'w', encoding='utf-8') as jf:
        json.dump({'total': total, 'valid': valid, 'invalid': invalid, 'files': rows}, jf, ensure_ascii=False, indent=2)

    print(f'Total files scanned: {total}')
    print(f'Valid PDFs: {valid}')
    print(f'Invalid (likely HTML/error): {invalid}')
    print(f'Report saved: {csvp} and {jsonp}')


if __name__ == '__main__':
    main()
