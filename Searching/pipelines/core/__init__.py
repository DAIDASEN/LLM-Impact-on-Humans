"""Core pipeline scripts packaged for safe imports.

These are copies of the top-level pipeline scripts adjusted so they can be
run as modules (e.g. `python -m pipelines.core.validate_paper`) while still
finding the repository-level `llm_pipeline` package.
"""

__all__ = [
    "validate_paper",
    "screen_cluster",
    "extract_summary",
    "ingest_arxiv_new",
]
