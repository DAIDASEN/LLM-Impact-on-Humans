import pandas as pd
import requests
from pathlib import Path

IN_CSV = "out_links/pdf_direct_links_with_title.csv"
  # 你下载的文件
OUT_DIR = Path("papers_direct")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_pdf_header(b: bytes) -> bool:
    return b.startswith(b"%PDF-")

df = pd.read_csv(IN_CSV)
ok, bad = 0, 0

for i, row in df.iterrows():
    url = str(row["pdf_url"]).strip()
    title = str(row.get("title", f"paper_{i}")).strip()
    fn = f"{i:04d}.pdf"
    path = OUT_DIR / fn

    if path.exists():
        continue

    try:
        r = requests.get(url, timeout=60, stream=True, allow_redirects=True)
        r.raise_for_status()

        # 先读一点点，验证是不是PDF
        head = b""
        it = r.iter_content(chunk_size=4096)
        head = next(it, b"")
        if not is_pdf_header(head):
            bad += 1
            print(f"[NOT PDF] {i} {url}")
            continue

        # 写文件
        with open(path, "wb") as f:
            f.write(head)
            for chunk in it:
                if chunk:
                    f.write(chunk)

        ok += 1
        print(f"[OK] {i} -> {fn}")

    except Exception as e:
        bad += 1
        print(f"[FAIL] {i} {url} ({e})")

print(f"Done. OK={ok}, BAD={bad}")
