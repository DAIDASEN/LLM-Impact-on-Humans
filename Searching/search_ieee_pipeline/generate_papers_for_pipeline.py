import pandas as pd
from pathlib import Path

src = Path('papers.csv')
dst = Path('papers_for_pipeline.csv')
papers_dir = Path('papers')

df = pd.read_csv(src)
available = set(p.name for p in papers_dir.glob('*.pdf'))
df2 = df[df['filename'].isin(available)].copy()
df2.to_csv(dst, index=False, encoding='utf-8-sig')
print(f'Wrote {len(df2)} rows to {dst}')
