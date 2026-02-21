from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NewsItem:
    ts: str
    headline: str
    source: str | None = None
    url: str | None = None


def fetch_finviz_news(symbol: str, *, limit: int = 8, timeout_s: int = 6) -> List[NewsItem]:
    """Fetch latest headlines from Finviz quote page.

    - No API key.
    - Best-effort parsing; returns [] on failures.
    """
    sym = (symbol or "").strip().upper()
    if not sym or not re.fullmatch(r"[A-Z0-9.\-]{1,10}", sym):
        return []

    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return []

    url = f"https://finviz.com/quote.ashx?t={sym}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    try:
        r = requests.get(url, headers=headers, timeout=timeout_s)
        if r.status_code != 200 or not r.text:
            return []
    except Exception:
        return []

    try:
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", class_="news-table")
        if not table:
            return []

        items: List[NewsItem] = []
        rows = table.find_all("tr")
        last_ts: Optional[str] = None
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            ts = cols[0].get_text(" ", strip=True)
            if ts:
                last_ts = ts
            else:
                ts = last_ts or ""

            a = cols[1].find("a")
            headline = a.get_text(" ", strip=True) if a else cols[1].get_text(" ", strip=True)
            href = a.get("href") if a else None

            src_span = cols[1].find("span")
            source = src_span.get_text(" ", strip=True) if src_span else None

            if headline:
                items.append(NewsItem(ts=ts, headline=headline, source=source, url=href))
            if len(items) >= max(1, int(limit)):
                break

        return items
    except Exception:
        return []
