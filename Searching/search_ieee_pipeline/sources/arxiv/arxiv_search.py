from ..search_arxiv import run as run_arxiv_search

def main(max_per_query=50, download=False, pause=3.0):
    run_arxiv_search(max_per_query, pause, download, out_dir='arxiv_pipeline_outputs')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max', type=int, default=50)
    p.add_argument('--download', action='store_true')
    p.add_argument('--pause', type=float, default=3.0)
    args = p.parse_args()
    main(max_per_query=args.max, download=args.download, pause=args.pause)
