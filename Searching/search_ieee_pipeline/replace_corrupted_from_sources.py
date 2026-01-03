#!/usr/bin/env python3
"""Replace corrupted PDFs in `papers/` with good copies from `ieee_pipeline_outputs/` when available."""
import pathlib
import shutil
import csv


PAPERS = pathlib.Path('papers')
SRC_ROOT = pathlib.Path('ieee_pipeline_outputs')
MIN_SIZE = 2000


def is_valid_pdf(p: pathlib.Path) -> bool:
    try:
        if not p.exists() or p.stat().st_size < MIN_SIZE:
            return False
        with open(p, 'rb') as f:
            return f.read(4) == b'%PDF'
    except Exception:
        return False


def find_in_src(name: str):
    for p in SRC_ROOT.rglob(name):
        return p
    return None


def load_mappings():
    # build map of basename -> full path in ieee_pipeline_outputs
    m = {}
    for csvp in SRC_ROOT.rglob('download_mapping.csv'):
        try:
            with open(csvp, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                hdr = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        requested, saved, url = row[0], row[1], row[2]
                        # saved may be relative path under cluster dir; try to find exact file
                        saved_basename = pathlib.Path(saved).name
                        # search for saved file under same folder as csv
                        candidate = (csvp.parent / saved_basename)
                        if candidate.exists():
                            m[requested.strip()] = candidate
                            m[saved_basename] = candidate
                        else:
                            # fallback: try rglob
                            found = next(SRC_ROOT.rglob(saved_basename), None)
                            if found:
                                m[requested.strip()] = found
                                m[saved_basename] = found
        except Exception:
            continue
    return m


def main():
    if not PAPERS.exists():
        print('papers/ not found')
        return
    corrupted = [p for p in PAPERS.rglob('*.pdf') if not is_valid_pdf(p)]
    print(f'Found {len(corrupted)} corrupted PDFs in papers/')
    mappings = load_mappings()
    replaced = 0
    for p in corrupted:
        key = p.name
        src = mappings.get(key)
        if not src:
            # try requested name matching without extension
            key_noext = p.stem
            src = mappings.get(key_noext)
        if src and is_valid_pdf(src):
            try:
                shutil.copy2(src, p)
                replaced += 1
                print(f'Replaced {p.name} from {src}')
            except Exception as e:
                print('failed to copy', p, e)
    print(f'Replaced {replaced} files out of {len(corrupted)})')


if __name__ == '__main__':
    main()
