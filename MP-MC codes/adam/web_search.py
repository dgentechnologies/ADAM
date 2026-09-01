"""
web_search.py — ADAM v40 DuckDuckGo web search
==============================================================================
Async wrapper around DuckDuckGo (ddgs / duckduckgo_search) with:
  • date-tagged results so the model can judge recency,
  • a short-lived per-query cache (SEARCH_CACHE_TTL) to avoid repeat lookups,
  • a minimum inter-request gap (SEARCH_MIN_GAP_S) to stay polite to DDG.

The heavy DDGS import is done here (optional — DDGS stays None if the package
isn't installed, and search degrades gracefully). main.py imports DDGS from
this module purely for its startup banner.
"""

import time
import asyncio
import datetime
import warnings

from config import SEARCH_CACHE_TTL, SEARCH_MIN_GAP_S

DDGS = None
try:
    try:
        from ddgs import DDGS as _D; DDGS = _D
    except ImportError:
        from duckduckgo_search import DDGS as _D; DDGS = _D
    print("✅ DuckDuckGo search ready")
except Exception as e:
    print(f"⚠️  DDG search unavailable: {e}")

_ddg_cache: dict = {}
_last_ddg_t      = 0.0


async def web_search(query: str, max_results: int = 4,
                     recent_only: bool = False) -> str:
    """
    DuckDuckGo search with current-date awareness.

    Two separate problems this addresses:
      1. The model itself has no innate sense of "today" beyond whatever
         training data cutoff it has — without being told the real date,
         it can construct stale-feeling queries or misjudge whether a
         search result is current. We prefix every query context (in the
         returned text) with today's actual date so the model can reason
         about recency correctly when it reads the results.
      2. DDG's own `timelimit` parameter (unused previously) can restrict
         results to a recent window server-side — this matters for
         anything genuinely time-sensitive (scores, news, "is X still
         happening") where a top-ranked but months-old result would
         otherwise be indistinguishable from a fresh one.
    """
    global _last_ddg_t
    if DDGS is None:
        return "Web search not available."
    q   = query.strip().lower()
    now = time.time()
    cache_key = f"{q}|{recent_only}"
    if cache_key in _ddg_cache:
        text, ts = _ddg_cache[cache_key]
        if now - ts < SEARCH_CACHE_TTL:
            return text
    gap = now - _last_ddg_t
    if gap < SEARCH_MIN_GAP_S:
        await asyncio.sleep(SEARCH_MIN_GAP_S - gap)
    try:
        def _run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # timelimit: "d"=past day, "w"=past week, "m"=past month.
                # Only applied when the caller signals this is a
                # time-sensitive query — a generic factual lookup
                # ("how does X work") shouldn't be needlessly restricted
                # to only very recent pages that may not exist.
                kwargs = {"max_results": max_results}
                if recent_only:
                    kwargs["timelimit"] = "m"
                return list(DDGS().text(query, **kwargs))
        results = await asyncio.to_thread(_run)
    except Exception as e:
        return f"Search failed: {e}"
    finally:
        _last_ddg_t = time.time()
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = str(r.get("title") or "").strip()
        body  = str(r.get("body") or r.get("snippet") or "").strip()
        if title or body:
            lines.append(f"• {title}: {body}" if title else f"• {body}")
    today_str = datetime.datetime.now().strftime("%A, %d %B %Y")
    text = (f"[Search performed on: {today_str}. Use this to judge "
            f"whether results below are current or outdated.]\n"
            + "\n".join(lines))
    _ddg_cache[cache_key] = (text, time.time())
    return text
