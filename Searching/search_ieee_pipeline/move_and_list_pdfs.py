import pathlib, csv

P=pathlib.Path('papers')
BAD=pathlib.Path('papers_bad')
BAD.mkdir(exist_ok=True)
MIN=2000
valid_files=[]
for p in P.rglob('*.pdf'):
    try:
        ok = p.stat().st_size>=MIN and p.read_bytes()[:4]==b'%PDF'
    except Exception:
        ok=False
    if not ok:
        dest=BAD/p.name
        try:
            p.replace(dest)
        except Exception:
            try:
                p.unlink()
            except Exception:
                pass
    else:
        valid_files.append(p.name)
# write papers_valid.csv
with open('papers_valid.csv','w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f)
    writer.writerow(['filename'])
    for n in valid_files:
        writer.writerow([n])
print('moved corrupted to', BAD)
print('valid count', len(valid_files))
