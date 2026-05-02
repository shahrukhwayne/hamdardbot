"""
scraper.py — Real-time web scraper for Jamia Hamdard University.

Official website: https://www.jamiahamdard.ac.in/

Verified URLs (live-checked April 2026):
  Admissions : https://www.jamiahamdard.ac.in/admissionnotice
  News        : https://www.jamiahamdard.ac.in/news
  Notices     : https://www.jamiahamdard.ac.in/officeorders
  Homepage    : https://www.jamiahamdard.ac.in/   (ticker / marquee links)

WHY THE PREVIOUS SCRAPER WAS WRONG:
  - BASE_URL was "https://jamiahamdard.edu" — this domain does NOT exist
    (the real domain is jamiahamdard.ac.in, an Indian academic TLD).
  - Endpoint paths were guessed (/admissions, /notices, /news-events) and
    none of them exist on the real site.
  - CSS selectors (.news-item, .notice-board, .admission-notice, etc.) were
    fabricated — none match the actual HTML structure of the site.
  - All three functions always hit the fallback data as a result.

Features (unchanged):
  - In-memory TTL cache (default 10 minutes)
  - Retry logic with exponential back-off (×2 per attempt)
  - Layered selector strategy + last-resort link/paragraph extraction
  - Structured return dicts for clean API consumption
"""

import re
import requests
from bs4 import BeautifulSoup
import time
import logging
import hashlib
from typing import Optional
from datetime import datetime

logger = logging.getLogger("scraper")

# ─────────────────────────────────────────────
#  CONFIG — OFFICIAL DOMAIN ONLY
# ─────────────────────────────────────────────

BASE_URL = "https://www.jamiahamdard.ac.in"          # ← FIXED (was jamiahamdard.edu)

# Verified page paths (live-checked against real site)
URL_ADMISSIONS = f"{BASE_URL}/admissionnotice"        # ← FIXED (was /admissions, /admission)
URL_NEWS       = f"{BASE_URL}/news"                   # ← FIXED (was /news-events)
URL_NOTICES    = f"{BASE_URL}/officeorders"           # ← FIXED (was /notices)
URL_HOME       = f"{BASE_URL}/"                       # homepage ticker for extra fallback

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_TIMEOUT = 12
MAX_RETRIES     = 3
RETRY_DELAY     = 2      # seconds; doubles on each retry
CACHE_TTL       = 600    # 10 minutes


# ─────────────────────────────────────────────
#  IN-MEMORY TTL CACHE  (unchanged interface)
# ─────────────────────────────────────────────

_cache: dict = {}


def _cache_key(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    hk = _cache_key(key)
    if hk in _cache:
        entry = _cache[hk]
        if time.time() - entry["ts"] < CACHE_TTL:
            logger.debug(f"[Cache HIT] {key}")
            return entry["data"]
        del _cache[hk]
    return None


def _cache_set(key: str, data: dict):
    hk = _cache_key(key)
    _cache[hk] = {"data": data, "ts": time.time()}


def invalidate_cache():
    """Clear all cached data (for testing or forced refresh)."""
    _cache.clear()
    logger.info("[Cache] Cleared.")


# ─────────────────────────────────────────────
#  CORE HTTP FETCHER  (retry + back-off, unchanged)
# ─────────────────────────────────────────────

def _fetch_page(url: str) -> Optional[BeautifulSoup]:
    """
    Fetch URL with exponential-back-off retry.
    Returns BeautifulSoup on success, None on total failure.
    Raw HTML is cached at the URL level to avoid duplicate downloads.
    """
    cached = _cache_get(url)
    if cached and "raw_html" in cached:
        logger.debug(f"[Cache] Returning cached HTML for {url}")
        return BeautifulSoup(cached["raw_html"], "html.parser")

    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[Scraper] Attempt {attempt}/{MAX_RETRIES}: {url}")
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=False)
            resp.raise_for_status()
            _cache_set(url, {"raw_html": resp.text})
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Scraper] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    logger.error(f"[Scraper] All {MAX_RETRIES} attempts failed for {url}")
    return None


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _is_useful(text: str, min_len: int = 15) -> bool:
    """Filter out nav labels, single words, and very short strings."""
    if len(text) < min_len:
        return False
    # Reject pure navigation noise
    noise = {"home", "about", "contact", "careers", "tenders", "library",
             "search", "login", "portal", "menu", "read more", "click here"}
    if text.lower() in noise:
        return False
    return True


def _links_from(soup: BeautifulSoup, within_selector: str, max_items: int) -> list:
    """
    Extract anchor text from <a> tags inside a CSS-selected container.
    Falls back gracefully if the selector matches nothing.
    """
    container = soup.select_one(within_selector)
    if not container:
        return []
    texts = []
    for a in container.find_all("a", href=True):
        t = _clean(a.get_text())
        if _is_useful(t) and t not in texts:
            texts.append(t)
        if len(texts) >= max_items:
            break
    return texts


def _extract_list_items(soup: BeautifulSoup, selectors: list, max_items: int = 5) -> list:
    """
    Try CSS selectors in priority order.
    Return up to max_items clean, useful strings.
    """
    for selector in selectors:
        elements = soup.select(selector)
        if not elements:
            continue
        texts = [_clean(el.get_text(separator=" ")) for el in elements]
        texts = [t for t in texts if _is_useful(t)][:max_items]
        if texts:
            logger.debug(f"[Scraper] Selector matched: '{selector}' → {len(texts)} items")
            return texts
    return []


# ─────────────────────────────────────────────
#  PUBLIC SCRAPING FUNCTIONS
# ─────────────────────────────────────────────

def get_admissions(max_items: int = 5) -> dict:
    """
    Scrape live admission notices from:
      https://www.jamiahamdard.ac.in/admissionnotice

    The page renders a <ul> list of notice links under an <h2> heading.
    Selector chain (priority order):
      1. h2 + ul  li  a          — primary: the notice bullet list
      2. .col-md-8 ul li         — content column list items
      3. .container ul li a      — broad container fallback
      4. Last resort: all <a> in main content with >15 chars
    """
    cache_key = f"admissions_{max_items}"
    cached = _cache_get(cache_key)
    if cached and "items" in cached:
        cached["from_cache"] = True
        return cached

    soup = _fetch_page(URL_ADMISSIONS)
    items = []

    if soup:
        # Strategy 1 — the verified notice <ul> list on the admissionnotice page
        selectors = [
            "h2 + ul li",          # <ul> immediately after the "Admission Notices" <h2>
            "h2 ~ ul li",          # <ul> anywhere after an <h2> (sibling)
            ".col-md-8 ul li",     # Bootstrap content column
            ".col-lg-8 ul li",
            "article ul li",
            ".content-area ul li",
            "main ul li",
        ]
        items = _extract_list_items(soup, selectors, max_items)

        # Strategy 2 — extract anchor text from the main content block
        if not items:
            items = _links_from(soup, ".col-md-8", max_items)

        # Strategy 3 — all anchors in body, exclude nav links (href contains keywords)
        if not items:
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                # Skip pure nav / file links
                if any(x in href for x in ["/uploads/files/", "ums.jamiahamdard", "#"]):
                    t = _clean(a.get_text())
                    if _is_useful(t, min_len=20) and t not in items:
                        items.append(t)
                if len(items) >= max_items:
                    break

    result = {
        "items": items if items else _fallback_admissions(),
        "source": URL_ADMISSIONS,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "from_cache": False,
        "fallback_used": not bool(items),
    }
    _cache_set(cache_key, result)
    return result


def get_news(max_items: int = 5) -> dict:
    """
    Scrape latest news from:
      https://www.jamiahamdard.ac.in/news   ("JH in News" page)

    The page contains news article cards / headings.
    Selector chain (priority order):
      1. article h3 / h4         — article card titles
      2. .news-content h3        — news content block
      3. .col-md-8 h3            — content column headings
      4. main h3, main h4        — broad heading fallback
      5. Homepage event cards    — fallback to home page event section
    """
    cache_key = f"news_{max_items}"
    cached = _cache_get(cache_key)
    if cached and "items" in cached:
        cached["from_cache"] = True
        return cached

    soup = _fetch_page(URL_NEWS)
    items = []

    if soup:
        selectors = [
            "article h3",
            "article h4",
            ".news-item h3",
            ".news-item h4",
            ".col-md-8 h3",
            ".col-lg-8 h3",
            "main h3",
            "main h4",
            ".container h3",
        ]
        items = _extract_list_items(soup, selectors, max_items)

        # Fallback to link text in the content column
        if not items:
            items = _links_from(soup, ".col-md-8", max_items)

    # Second fallback: scrape homepage event section headings
    if not items:
        logger.info("[Scraper] News page yielded nothing — trying homepage events section")
        home_soup = _fetch_page(URL_HOME)
        if home_soup:
            selectors_home = [
                ".event-section h3",
                ".events h3",
                ".news-events h3",
                "section h3",
                "section h4",
            ]
            items = _extract_list_items(home_soup, selectors_home, max_items)

            # Last resort on homepage: <a> inside any ticker / marquee
            if not items:
                for tag in home_soup.find_all(["marquee", "div"],
                                              class_=re.compile(r"ticker|scroll|news", re.I)):
                    for a in tag.find_all("a"):
                        t = _clean(a.get_text())
                        if _is_useful(t, 20) and t not in items:
                            items.append(t)
                    if len(items) >= max_items:
                        break

    result = {
        "items": items if items else _fallback_news(),
        "source": URL_NEWS,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "from_cache": False,
        "fallback_used": not bool(items),
    }
    _cache_set(cache_key, result)
    return result


def get_notices(max_items: int = 5) -> dict:
    """
    Scrape official office orders / circulars from:
      https://www.jamiahamdard.ac.in/officeorders

    The page lists circulars as <a> links (PDF / page links) in a content div.
    Selector chain (priority order):
      1. .col-md-8 ul li a       — list of notice anchor links
      2. main ul li a            — broader list fallback
      3. table td a              — table-format notices
      4. .col-md-8 a             — any anchor in content column
    """
    cache_key = f"notices_{max_items}"
    cached = _cache_get(cache_key)
    if cached and "items" in cached:
        cached["from_cache"] = True
        return cached

    soup = _fetch_page(URL_NOTICES)
    items = []

    if soup:
        # Primary — list anchors in the content column
        selectors = [
            ".col-md-8 ul li a",
            ".col-lg-8 ul li a",
            "article ul li a",
            "main ul li a",
            "table td a",
            ".col-md-8 a",
            "main a",
        ]
        items = _extract_list_items(soup, selectors, max_items)

        # If selectors returned container text (includes URL noise), re-extract cleanly
        if not items:
            content = soup.select_one(".col-md-8") or soup.select_one("main")
            if content:
                for a in content.find_all("a", href=True):
                    t = _clean(a.get_text())
                    if _is_useful(t, 20) and t not in items:
                        items.append(t)
                    if len(items) >= max_items:
                        break

    result = {
        "items": items if items else _fallback_notices(),
        "source": URL_NOTICES,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "from_cache": False,
        "fallback_used": not bool(items),
    }
    _cache_set(cache_key, result)
    return result


# ─────────────────────────────────────────────
#  STATIC FALLBACKS  (accurate 2026 data)
# ─────────────────────────────────────────────

def _fallback_admissions() -> list:
    return [
        "Admissions Open 2026–27: UG & PG applications live at ums.jamiahamdard.ac.in",
        "Last date for UG & PG courses (except MBA & M.Sc Nursing): 31st May 2026.",
        "MBA Interview dates notice published — check admissionnotice page.",
        "M.Sc Nursing Entrance Test rescheduled to 18th April 2026.",
        "PhD Cycle-II (2025–26) entrance test on 1st May 2026; last date was 15th April.",
        "For admission queries: 7042519957 | admissions@jamiahamdard.ac.in",
    ]


def _fallback_news() -> list:
    return [
        "15th Convocation of Jamia Hamdard scheduled on 16th April 2026.",
        "Techno-Cultural Fest celebrated by School of Engineering, Sciences and Technology.",
        "LOI signed between Jamia Hamdard and Fathima Healthcare for Dubai off-campus.",
        "National Science Day 2026 celebrated with theme 'Women in Science: Catalysing Viksit Bharat'.",
        "Capacity Building Workshop: International Day for Girls in ICT on 24th April 2026.",
        "PM VidyaLakshmi Loan Scheme information now available on official portal.",
    ]


def _fallback_notices() -> list:
    return [
        "Important Notice to Students regarding 15th Convocation on 16th April 2026.",
        "Prospectus 2026-27 published — download from official website.",
        "Admissions open for all UG & PG programmes at ums.jamiahamdard.ac.in",
        "Viksit Bharat Youth Parliament 2025: Registration deadline passed.",
        "CCFIS course admissions open — check uploads/files/CCIFS.jpeg for details.",
    ]


# ─────────────────────────────────────────────
#  FORMAT HELPER  (unchanged interface)
# ─────────────────────────────────────────────

def format_scraped_response(data: dict, label: str = "Here is the latest information") -> str:
    """Convert a scraper result dict into a readable chat response string."""
    if not data.get("items"):
        return (
            f"I'm unable to fetch live data right now. "
            f"Please visit {data.get('source', BASE_URL)} for the latest information."
        )

    lines = [f"📋 **{label}** _(as of {data['fetched_at']}):_\n"]
    for i, item in enumerate(data["items"], 1):
        lines.append(f"{i}. {item}")

    if data.get("fallback_used"):
        lines.append("\n_⚠️ Note: Could not reach the live website. Showing last-known data._")
    else:
        lines.append(f"\n_🌐 Live source: {data['source']}_")

    return "\n".join(lines)