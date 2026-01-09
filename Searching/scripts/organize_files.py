import shutil
from pathlib import Path
import csv
import datetime
import sys


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_csvs(repo: Path, dst_csv_dir: Path, archive_dir: Path):
    copied = []
    for p in repo.rglob("*.csv"):
        # skip files under data/, archive/, outputs/, llm_outputs/, scripts/
        if any(part in ("data", "archive", "outputs", "llm_outputs", "scripts") for part in p.parts):
            continue
        rel = p.relative_to(repo)
        target = dst_csv_dir / rel.name
        shutil.copy2(p, target)
        # also copy to archive
        archive_target = archive_dir / rel.name
        shutil.copy2(p, archive_target)
        copied.append(str(rel))
    return copied


def load_mapping(repo: Path):
    mfile = repo / "filename_origin_map.csv"
    mapping = {}
    if not mfile.exists():
        print("Mapping file not found:", mfile)
        return mapping
    with mfile.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mapping[r["filename"].strip()] = r
    return mapping


def move_pdfs(repo: Path, mapping: dict, dst_arxiv: Path, dst_gs: Path):
    moved = []
    missing = []
    # find candidate PDF files in repo (excluding data/archive/outputs)
    candidates = list(repo.rglob("*.pdf"))
    index = {}
    for p in candidates:
        if any(part in ("data", "archive", "outputs", "llm_outputs") for part in p.parts):
            continue
        index.setdefault(p.name, []).append(p)

    for fn, meta in mapping.items():
        origin = meta.get("origin", "").strip()
        targets = index.get(fn)
        if not targets:
            missing.append(fn)
            continue
        # if multiple, pick first
        src = targets[0]
        if origin == "arxiv":
            dst = dst_arxiv / fn
        elif origin == "google_scholar":
            dst = dst_gs / fn
        else:
            # place unknown into data/papers/other
            dst = repo / "data" / "papers" / "other" / fn
            ensure_dir(dst.parent)
        ensure_dir(dst.parent)
        try:
            shutil.move(str(src), str(dst))
            moved.append((str(src), str(dst), origin))
        except Exception as e:
            print(f"Failed moving {src} -> {dst}: {e}")
    return moved, missing


def main():
    repo = Path(__file__).resolve().parent.parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_backup = repo / "archive" / f"backup_{timestamp}"
    ensure_dir(archive_backup)

    dst_csv_dir = repo / "data" / "csv"
    ensure_dir(dst_csv_dir)

    dst_papers_arxiv = repo / "data" / "papers" / "arxiv"
    dst_papers_gs = repo / "data" / "papers" / "google_scholar"
    ensure_dir(dst_papers_arxiv)
    ensure_dir(dst_papers_gs)

    print("Copying CSVs to", dst_csv_dir)
    copied = copy_csvs(repo, dst_csv_dir, archive_backup)
    print(f"Copied {len(copied)} CSVs (examples):", copied[:10])

    print("Loading filename_origin_map.csv")
    mapping = load_mapping(repo)
    if not mapping:
        print("No mapping found; aborting PDF move.")
        sys.exit(1)

    print("Moving PDFs based on mapping...")
    moved, missing = move_pdfs(repo, mapping, dst_papers_arxiv, dst_papers_gs)
    print(f"Moved {len(moved)} PDFs; missing {len(missing)} files")
    if missing:
        print("Missing examples:", missing[:10])

    # write a small report
    report = archive_backup / "organize_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"csv_copied: {len(copied)}\n")
        f.write(f"pdf_moved: {len(moved)}\n")
        f.write(f"pdf_missing: {len(missing)}\n")
        f.write("moved_details:\n")
        for s, d, o in moved:
            f.write(f"{s} -> {d} ({o})\n")
        if missing:
            f.write("missing:\n")
            for m in missing:
                f.write(m + "\n")

    print("Report written to", report)


if __name__ == "__main__":
    main()
