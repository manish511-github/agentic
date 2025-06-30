"""Utility helpers for assembling Reddit search query strings in batches.

These helpers take a list of keywords and yield query strings joined by the
desired boolean operator.  Batching keeps query strings below Reddit's length
limits while maximising keyword coverage.
"""

from typing import Generator, Iterable

import structlog


logger = structlog.get_logger()


def _maybe_quote(keyword: str) -> str:
    """Return the keyword wrapped in double quotes **iff** it contains
    whitespace and is not already quoted.  Quoting preserves the multi-word
    phrase during Reddit search.
    """

    kw = keyword.strip()
    if kw.startswith("\"") and kw.endswith("\""):
        return kw  # Already quoted.
    if " " in kw:
        return f'"{kw}"'
    return kw


def _batch_keywords(keywords: Iterable[str], batch_size: int, quote: bool) -> Generator[list[str], None, None]:
    """Yield successive *batch_size* chunks from *keywords*.  Optionally quote
    each keyword via :func:`_maybe_quote`.
    """

    cur_batch = []
    for kw in keywords:
        cur_batch.append(_maybe_quote(kw) if quote else kw)
        if len(cur_batch) == batch_size:
            yield cur_batch
            cur_batch = []
    if cur_batch:
        yield cur_batch


def create_OR_query_in_batch(query_list: list[str], batch_size: int, *, quote: bool = False) -> Generator[str, None, None]:
    """Yield Reddit search query strings that join *batch_size* keywords with
    the ``OR`` Boolean operator.

    Parameters
    ----------
    query_list
        The keywords/phrases to combine.
    batch_size
        Maximum number of keywords per query string.
    quote
        If *True*, quote multi-word phrases automatically.
    """

    for batch in _batch_keywords(query_list, batch_size, quote):
        yield " OR ".join(batch)


def create_AND_query_in_batch(query_list: list[str], batch_size: int, *, quote: bool = False) -> Generator[str, None, None]:
    """Yield query strings joining keywords with ``AND``."""

    for batch in _batch_keywords(query_list, batch_size, quote):
        yield " AND ".join(batch)


def create_NOT_query_in_batch(query_list: list[str], batch_size: int, *, quote: bool = False) -> Generator[str, None, None]:
    """Yield query strings joining keywords with ``NOT``."""

    for batch in _batch_keywords(query_list, batch_size, quote):
        yield " NOT ".join(batch)