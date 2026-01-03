import pathlib

MIN=2000
root=pathlib.Path('ieee_pipeline_outputs')
valid=0
total=0
bad=[]
for p in root.rglob('*.pdf'):
    total+=1
    try:
        if p.stat().st_size>=MIN and p.read_bytes()[:4]==b'%PDF':
            valid+=1
        else:
            bad.append(p)
    except Exception:
        bad.append(p)
print('total',total,'valid',valid,'bad',len(bad))
for b in bad[:20]:
    print('BAD',b)
