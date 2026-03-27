"""
Papal Encyclicals Web Scraper
DS 5001 - Exploratory Text Analytics Final Project

Scrapes encyclical texts from https://www.papalencyclicals.net/document-directory
Collects metadata (pope, title, date, language) and full document text.

Usage:
    python src/scraper.py                  # Scrape everything
    python src/scraper.py --index-only     # Only scrape the directory index
    python src/scraper.py --max-docs 10    # Limit number of documents
"""

import os
import re
import csv
import json
import time
import warnings
import logging
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from bs4 import FeatureNotFound
from urllib3.exceptions import InsecureRequestWarning

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.papalencyclicals.net"
DIRECTORY_URL = f"{BASE_URL}/document-directory"

# Known dead/unreliable archival hosts. Keep metadata rows, skip fetch attempts.
KNOWN_DEAD_DOMAINS = {
    "digilander.iol.it",
    "digilander.libero.it",
    "cin.org",
    "www.cin.org",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_FILE = DATA_DIR / "encyclicals_index.json"
LIBRARY_FILE = DATA_DIR / "processed" / "LIBRARY.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 1.5  # seconds between requests (be polite)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# TLS behavior (secure-by-default):
# - SCRAPER_CA_BUNDLE can point to a custom corporate/root CA bundle.
# - SCRAPER_ALLOW_INSECURE_SSL=true enables a last-resort fallback with verify=False.
def get_ca_bundle() -> str | None:
    return os.getenv("SCRAPER_CA_BUNDLE")


def allow_insecure_ssl() -> bool:
    return _env_flag("SCRAPER_ALLOW_INSECURE_SSL", default=False)


def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().strip()


_INSECURE_SSL_NOTICE_SHOWN = False
_EPUB_DEPENDENCY_NOTICE_SHOWN = False


def _prepare_insecure_ssl_mode():
    """Suppress repetitive urllib3 warnings when insecure SSL fallback is enabled."""
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    urllib3.disable_warnings(InsecureRequestWarning)


def _make_soup(html: str | bytes) -> BeautifulSoup:
    """Build BeautifulSoup with lxml when available; otherwise fall back safely.

    Accepts bytes so BeautifulSoup can auto-detect encoding from the page's
    own <meta charset> tag (avoids requests defaulting to ISO-8859-1).
    """
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Parse the document directory
# ---------------------------------------------------------------------------

def fetch_page(url: str, retries: int = 3) -> bytes | None:
    """Fetch a page with retries and polite delay.

    Returns raw bytes so callers can pass them directly to BeautifulSoup,
    which then reads the page's own <meta charset> for correct encoding.
    """
    global _INSECURE_SSL_NOTICE_SHOWN

    if _domain(url) in KNOWN_DEAD_DOMAINS:
        return None

    ca_bundle = get_ca_bundle()
    verify = ca_bundle or True
    if ca_bundle:
        logger.info(f"Using custom CA bundle from SCRAPER_CA_BUNDLE: {ca_bundle}")

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, verify=verify)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.content
        except requests.exceptions.SSLError as e:
            if allow_insecure_ssl():
                try:
                    _prepare_insecure_ssl_mode()
                    if not _INSECURE_SSL_NOTICE_SHOWN:
                        logger.info(
                            "SSL verification failed in this environment; "
                            "using insecure fallback (verify=False)."
                        )
                        _INSECURE_SSL_NOTICE_SHOWN = True
                    resp = requests.get(url, headers=HEADERS, timeout=30, verify=False)
                    resp.raise_for_status()
                    time.sleep(REQUEST_DELAY)
                    return resp.content
                except requests.RequestException as insecure_err:
                    logger.debug(f"Insecure SSL retry failed for {url}: {insecure_err}")
            else:
                logger.debug(f"Attempt {attempt+1} SSL failure for {url}: {e}")

            time.sleep(2 ** (attempt + 1))
        except requests.RequestException as e:
            logger.debug(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(2 ** (attempt + 1))
    logger.info(f"Failed to fetch {url} after {retries} attempts")
    return None


def parse_directory(html: str | bytes) -> list[dict]:
    """
    Parse the document directory page.

    The page organizes documents under pope headings. Each pope section
    contains links to individual encyclical pages. We extract:
      - pope name
      - document title
      - document URL
      - year (parsed from title or context when available)
    """
    soup = _make_soup(html)
    documents = []

    content = soup.find("div", class_="entry-content") or soup.find("article") or soup.find("main") or soup.body

    if content is None:
        logger.error("Could not find main content area on directory page")
        return documents

    section_count = 0
    current_section = None

    for node in content.descendants:
        node_name = getattr(node, "name", None)

        if node_name in {"h2", "h3", "h4"}:
            heading_text = node.get_text(" ", strip=True)
            section_meta = _parse_section_heading(heading_text)
            if section_meta is not None:
                current_section = section_meta
                section_count += 1
            continue

        if node_name != "a" or current_section is None:
            continue

        href = (node.get("href") or "").strip()
        title = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()

        if not href or not title:
            continue

        href_lower = href.lower()
        if any(skip in href_lower for skip in [
            "javascript:",
            "mailto:",
            "facebook",
            "twitter",
            "instagram",
            "youtube",
            "donate",
            "contact",
            "privacy",
            "comment",
            "reply",
            "login",
            "register",
        ]):
            continue

        # Skip category/index links that represent section headers, not documents.
        if re.search(r"/(?:category|tag)/", href_lower):
            continue

        if _is_non_document_link_title(title):
            continue

        if not href.startswith("http"):
            href = urljoin(BASE_URL, href)

        parsed = urlparse(href)
        if "papalencyclicals.net" not in parsed.netloc:
            # Keep external links only when they look like document records.
            if len(title) < 4:
                continue

        if href.rstrip("/") == DIRECTORY_URL.rstrip("/"):
            continue

        # Skip Vatican category/section index pages (e.g. homilies/index_en.htm,
        # doc_doc_index.htm, angelus.index.html).  These are navigation hubs,
        # not individual documents.
        if "vatican.va" in parsed.netloc.lower():
            filename = parsed.path.rstrip("/").rsplit("/", 1)[-1].lower()
            if "index" in filename and re.search(r"\.html?$", filename):
                continue

        # Skip links whose title looks like a section heading
        # (e.g. "Pope Benedict XVI \u2013 April 19, 2005 to February 28, 2013").
        if _parse_section_heading(title) is not None:
            continue

        if current_section["is_councils"] and "/councils/" not in parsed.path.lower():
            continue

        year = extract_year(title)
        doc_id = make_doc_id(current_section["name"], title)

        is_council = bool(current_section["is_councils"])
        category = "council" if is_council else "pope"
        author = title if is_council else current_section["name"]

        documents.append({
            "doc_id": doc_id,
            "category": category,
            "author": author,
            "author_dates": "" if is_council else current_section["dates"],
            "pope": current_section["name"],
            "pope_dates": current_section["dates"],
            "title": title,
            "url": href,
            "year": year,
            "document_type": infer_document_type(title, href),
            "language": "en",  # default; updated during text extraction
            "format": "",       # set during text extraction
        })

    # Deduplicate by URL
    seen_urls = set()
    unique_docs = []
    for doc in documents:
        if doc["url"] not in seen_urls:
            seen_urls.add(doc["url"])
            unique_docs.append(doc)

    for doc in unique_docs:
        if not doc.get("pope") or doc["pope"].strip().lower() == "unknown":
            doc["pope"] = "Church Councils" if "/councils/" in doc["url"].lower() else "Unclassified"

    logger.info(f"Parsed {len(unique_docs)} unique documents from {section_count} sections")
    return unique_docs


def _parse_section_heading(text: str) -> dict | None:
    """Return section metadata for pope headings and Church Councils."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return None

    lower = clean.lower()

    if "church councils" in lower:
        return {"name": "Church Councils", "dates": "", "is_councils": True}

    if not lower.startswith("pope"):
        return None

    years = re.findall(r"\b(1[2-9]\d{2}|20[0-2]\d)\b", clean)
    dates = ""
    if len(years) >= 2:
        dates = f"{years[0]}-{years[1]}"
    elif len(years) == 1:
        dates = years[0]

    _MONTH_PAT = re.compile(
        r"\b(January|February|March|April|May|June|July|August"
        r"|September|October|November|December"
        r"|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?"
        r"|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\b",
        re.IGNORECASE,
    )
    year_start = re.search(r"\b(1[2-9]\d{2}|20[0-2]\d)\b", clean)
    month_start = _MONTH_PAT.search(clean)
    cut = min(
        (m.start() for m in (month_start, year_start) if m),
        default=None,
    )
    if cut is not None:
        name = clean[:cut].rstrip(" ,;-–—")
    else:
        name = clean

    name = re.sub(r"\s+", " ", name).strip()
    # Some headings contain placeholder tokens like "?, ?" before date ranges.
    name = re.sub(r"\s*\?+\s*(,\s*\?+\s*)*$", "", name).strip()
    name = re.sub(r"[\s,;:\-–—]+$", "", name).strip()
    if not name:
        return None

    return {"name": name, "dates": dates, "is_councils": False}


def _is_non_document_link_title(title: str) -> bool:
    """Filter out language toggles and structural links that are not documents."""
    lower = title.strip().lower()
    if not lower:
        return True

    if lower in {
        "alphabetic order",
        "alphabetical order",
        "chronological order",
        "chronological orders",
        "encyclicals",
        "other writings",
        "index",
        "go",
        "menu",
        "search tips",
        "site map",
    }:
        return True

    if lower in {
        "english", "latin", "italian", "french", "german", "spanish",
        "portuguese", "polish", "chinese", "arabic", "dutch", "croatian",
        "hungarian", "czech", "byelorussian", "slovak", "slovenian",
        "romanian", "ukrainian", "vietnamese", "korean", "japanese",
        "traditional chinese", "simplified chinese",
        "mobi", "epub", "pdf",
        "several other languages", "several languages", "other languages",
        "multiple languages",
    }:
        return True

    # Catch any remaining "<qualifier> <language>" two-word language labels.
    if re.match(r"^(traditional|simplified|old|ancient|modern|classical)\s+\w+$", lower):
        return True

    return False


def extract_year(text: str) -> str:
    """Try to extract a 4-digit year from text."""
    match = re.search(r"\b(1[2-9]\d{2}|20[0-2]\d)\b", text)
    return match.group(1) if match else ""


def make_doc_id(pope: str, title: str) -> str:
    """Create a clean document ID from pope and title."""
    pope_clean = re.sub(r"[^\w]", "_", pope.strip())[:30]
    title_clean = re.sub(r"[^\w]", "_", title.strip())[:50]
    return f"{pope_clean}__{title_clean}".lower()


def infer_document_type(title: str, url: str = "", text: str = "") -> str:
    """Infer a normalized document type from title, URL, and optional text."""
    title_blob = f"{title} {text[:1500]}".lower()
    url_blob = (url or "").lower()
    blob = f"{title_blob} {url_blob}"

    patterns = [
        ("index", ["index", ".index.html", "alphabetic order", "chronological order"]),
        ("encyclical", [" enc_", "/encyclicals/", "encyclical"]),
        ("apostolic exhortation", ["apostolic exhortation", "exhortation", " exh_", "apost_exhortations"]),
        ("apostolic constitution", ["apostolic constitution", "apostolic constitutions", "apc_", "apost_constitutions"]),
        ("apostolic letter", ["apostolic letter", "apostolic letters", "letters", " let_"]),
        ("motu proprio", ["motu proprio", "motu-proprio", "moto proprio"]),
        ("homily", ["homily", "homilies", " hom_"]),
        ("speech", ["speech", "speeches", " spe_"]),
        ("message", ["message", "messages", "msg", "communications day", "world day of peace"]),
        ("angelus", ["angelus", "regina coeli"]),
        ("audience", ["audience", "audiences"]),
        ("bull", ["[bull]", " bolla ", " papal bull "]),
        ("decree", ["decree", "decret"]),
        ("brief", [" brief ", " breve "]),
        ("instruction", ["instruction"]),
        ("prayer", ["prayer", "preghiera"]),
        ("travel", ["travel", "viaggio"]),
        ("biography", ["biography", "bio"]),
    ]

    for doc_type, hints in patterns:
        if doc_type == "encyclical":
            # Avoid false positives from the papalencyclicals.net domain name.
            if " enc_" in blob or "/encyclicals/" in url_blob or "encyclical" in title_blob:
                return doc_type
            continue

        if any(h in blob for h in hints):
            return doc_type

    return "document"


def _find_content_area(soup: BeautifulSoup):
    """Return the best non-empty content container from a parsed page."""
    def _has_text(tag) -> bool:
        return tag is not None and len(tag.get_text(strip=True)) > 200

    return next(
        (c for c in [
            soup.find("div", class_="entry-content"),
            soup.find("article"),
            soup.find("div", class_="post-content"),
            soup.find("main"),
            soup.find("div", class_="content"),
            soup.find("div", id="content"),
        ] if _has_text(c)),
        soup.body,
    )


# Matches a month name (English / Italian / French / Latin) followed within
# 20 characters by a 4-digit year.  Day numbers before the month are fine
# because the regex anchors on the month word, not the day.
_DATE_RE = re.compile(
    r"\b(?:"
    # English full
    r"January|February|March|April|May|June|July|August"
    r"|September|October|November|December"
    # English abbreviated
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    # Italian
    r"|gennaio|febbraio|marzo|aprile|maggio|giugno"
    r"|luglio|agosto|settembre|ottobre|novembre|dicembre"
    # French
    r"|janvier|f\xe9vrier|mars|avril|mai|juin"
    r"|juillet|ao\xfbt|septembre|octobre|novembre|d\xe9cembre"
    # Latin (genitive forms common in papal documents)
    r"|ianuarii|februarii|martii|aprilis|maii|iunii"
    r"|iulii|augusti|septembris|octobris|novembris|decembris"
    r")\b.{0,20}?\b(1[2-9]\d{2}|20[0-2]\d)\b",
    re.IGNORECASE | re.DOTALL,
)
_YEAR_RE = re.compile(r"\b(1[2-9]\d{2}|20[0-2]\d)\b")


def _extract_year_from_page(
    soup: BeautifulSoup,
    valid_range: tuple[int, int] | None = None,
) -> str:
    """
    Try to find the publication year on a document page.

    Searches the main content area only (avoids nav/footer metadata dates).
    If `valid_range` is provided as (min_year, max_year), only years within
    that window (plus a ±50-year tolerance) are accepted.
    """
    def _plausible(year_str: str) -> bool:
        if not valid_range:
            return True
        y = int(year_str)
        lo, hi = valid_range
        return (lo - 50) <= y <= (hi + 50)

    # Restrict search to main content area to avoid nav/sidebar/footer dates
    content = _find_content_area(soup)
    if content is None:
        return ""

    tags = content.find_all(["h1", "h2", "h3", "h4", "p", "b", "strong"])

    # Pass 0: search the LAST few tags first — papal subscription lines
    # ("Datum Romae... die X mensis Y anno Z") are always at the document end.
    for tag in reversed(tags[-10:]):
        text = tag.get_text(" ", strip=True)
        m = _DATE_RE.search(text)
        if m and _plausible(m.group(1)):
            return m.group(1)

    # Pass 1: month + year, full document order
    for tag in tags:
        text = tag.get_text(" ", strip=True)
        m = _DATE_RE.search(text)
        if m and _plausible(m.group(1)):
            return m.group(1)

    # Pass 2: standalone 4-digit year within valid range, still tag-based
    if valid_range:
        for tag in tags:
            text = tag.get_text(" ", strip=True)
            for m in _YEAR_RE.finditer(text):
                if _plausible(m.group(1)):
                    return m.group(1)

    # Pass 3: scan the full content text (catches years in plain text nodes /
    # divs not covered by the tag search above)
    full_text = content.get_text(" ", strip=True)
    for m in _DATE_RE.finditer(full_text):
        if _plausible(m.group(1)):
            return m.group(1)
    if valid_range:
        for m in _YEAR_RE.finditer(full_text):
            if _plausible(m.group(1)):
                return m.group(1)

    return ""


# ---------------------------------------------------------------------------
# Step 2: Extract text from individual document pages
# ---------------------------------------------------------------------------

def _extract_year_from_text(text: str, valid_range: tuple[int, int] | None = None) -> str:
    """
    Extract a publication year from plain (already-extracted) document text.
    Same logic as _extract_year_from_page but operates on a plain string.
    """
    def _plausible(y: str) -> bool:
        if not valid_range:
            return True
        lo, hi = valid_range
        return (lo - 50) <= int(y) <= (hi + 50)

    # Pass 0: search the last 1000 chars first — subscription date is always
    # at the end of the document.
    tail = text[-1000:] if len(text) > 1000 else text
    for m in _DATE_RE.finditer(tail):
        if _plausible(m.group(1)):
            return m.group(1)

    # Pass 1: month + year, full document
    for m in _DATE_RE.finditer(text):
        if _plausible(m.group(1)):
            return m.group(1)

    # Pass 2: standalone year within valid range
    if valid_range:
        for m in _YEAR_RE.finditer(text):
            if _plausible(m.group(1)):
                return m.group(1)

    return ""


def _parse_pope_dates(pope_dates: str) -> tuple[int, int] | None:
    """Parse a pope_dates string like '1334-1342' into (min_year, max_year)."""
    if not pope_dates:
        return None
    years = re.findall(r"\b(1[2-9]\d{2}|20[0-2]\d)\b", pope_dates)
    if not years:
        return None
    ints = [int(y) for y in years]
    return (min(ints), max(ints))


def extract_document_text(url: str, pope_dates: str = "") -> dict:
    """
    Fetch and extract text from an individual encyclical page.

    Returns dict with keys: text, format, language, year, document_type, error
    """
    html = fetch_page(url)
    if html is None:
        return {"text": "", "format": "error", "language": "", "year": "", "error": "fetch_failed"}

    soup = _make_soup(html)

    # Check if the page has an epub/pdf download link
    epub_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if href.endswith(".epub"):
            epub_link = urljoin(url, a["href"])
            break

    # Try to extract text from the HTML page itself
    text = extract_text_from_html(soup)

    # If we got very little text, try the epub
    if len(text.strip()) < 200 and epub_link:
        epub_text = download_and_extract_epub(epub_link)
        if epub_text and len(epub_text) > len(text):
            text = epub_text
            fmt = "epub"
        else:
            fmt = "html"
    else:
        fmt = "html"

    # Detect language (simple heuristic)
    language = detect_language_simple(text)
    valid_range = _parse_pope_dates(pope_dates)
    year = _extract_year_from_page(soup, valid_range=valid_range)

    document_type = infer_document_type(soup.title.get_text(" ", strip=True) if soup.title else "", url, text)

    return {
        "text": text.strip(),
        "format": fmt,
        "language": language,
        "year": year,
        "document_type": document_type,
        "error": "" if text.strip() else "no_text_extracted",
    }


def extract_text_from_html(soup: BeautifulSoup) -> str:
    """Extract the main document text from an HTML page."""
    # Remove script, style, nav, header, footer elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer",
                               "aside", "form", "iframe"]):
        tag.decompose()

    # Try common content containers in priority order, skipping any that are
    # effectively empty (Vatican.va and others have a div.content shell with
    # the real text inside <main> instead).
    content = _find_content_area(soup)

    if content is None:
        # Fall back to body
        content = soup.body

    if content is None:
        return ""

    # Get text, preserving paragraph breaks
    paragraphs = []
    for p in content.find_all(["p", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    if paragraphs:
        return "\n\n".join(paragraphs)

    # If no paragraphs found, get all text
    return content.get_text(separator="\n", strip=True)


def download_and_extract_epub(epub_url: str) -> str:
    """Download an epub file and extract its text content."""
    global _EPUB_DEPENDENCY_NOTICE_SHOWN

    try:
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            if not _EPUB_DEPENDENCY_NOTICE_SHOWN:
                logger.info(
                    "ebooklib is not installed; EPUB extraction is disabled. "
                    "Install with: pip install ebooklib"
                )
                _EPUB_DEPENDENCY_NOTICE_SHOWN = True
            return ""

        import tempfile

        verify = get_ca_bundle() or True

        try:
            resp = requests.get(epub_url, headers=HEADERS, timeout=60, verify=verify)
        except requests.exceptions.SSLError:
            if not allow_insecure_ssl():
                raise
            logger.warning(
                "EPUB download hit SSL verification failure; retrying with SSL verification disabled."
            )
            _prepare_insecure_ssl_mode()
            resp = requests.get(epub_url, headers=HEADERS, timeout=60, verify=False)

        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
            f.write(resp.content)
            temp_path = f.name

        try:
            book = epub.read_epub(temp_path)
            texts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = _make_soup(item.get_content())
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    texts.append(text)
            return "\n\n".join(texts)
        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.warning(f"Failed to extract epub from {epub_url}: {e}")
        return ""


def detect_language_simple(text: str) -> str:
    """Simple language detection based on common words."""
    if not text:
        return "unknown"

    text_lower = text[:2000].lower()

    # Count common words for a few languages
    en_words = ["the", "and", "of", "to", "in", "that", "which", "for", "with", "this"]
    la_words = ["et", "in", "est", "non", "qui", "quae", "sed", "cum", "ad", "ut"]
    it_words = ["il", "della", "che", "dei", "nella", "alla", "sono", "questo", "quella", "anche"]
    fr_words = ["le", "la", "les", "des", "dans", "qui", "est", "pour", "avec", "nous"]

    scores = {
        "en": sum(1 for w in en_words if re.search(rf"\b{w}\b", text_lower)),
        "la": sum(1 for w in la_words if re.search(rf"\b{w}\b", text_lower)),
        "it": sum(1 for w in it_words if re.search(rf"\b{w}\b", text_lower)),
        "fr": sum(1 for w in fr_words if re.search(rf"\b{w}\b", text_lower)),
    }

    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "unknown"


# ---------------------------------------------------------------------------
# Step 3: Orchestrate scraping
# ---------------------------------------------------------------------------

def scrape_index() -> list[dict]:
    """Scrape the document directory and return the index."""
    logger.info(f"Fetching document directory from {DIRECTORY_URL}")
    html = fetch_page(DIRECTORY_URL)
    if html is None:
        raise RuntimeError("Failed to fetch the document directory page")

    documents = parse_directory(html)
    return documents


def scrape_documents(documents: list[dict], max_docs: int = 0,
                     english_only: bool = False,
                     show_progress: bool = False,
                     progress_label: str = "Scraping") -> list[dict]:
    """
    Scrape text for each document in the index.
    Saves raw text files to data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = len(documents)
    if max_docs > 0:
        documents = documents[:max_docs]
        logger.info(f"Limiting to {max_docs} of {total} documents")

    skipped = 0
    extracted = 0
    errors = 0
    dead_domain_skipped = 0

    def update_progress(idx: int):
        if not show_progress:
            return
        print(
            f"\r{progress_label}: {idx}/{len(documents)} | "
            f"saved={extracted} skipped={skipped} dead={dead_domain_skipped} errors={errors}",
            end="",
            flush=True,
        )

    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        raw_file = RAW_DIR / f"{doc_id}.txt"

        # Skip known dead domains to avoid repeated DNS/connection noise.
        if _domain(doc.get("url", "")) in KNOWN_DEAD_DOMAINS:
            doc["error"] = "known_dead_domain"
            doc["text_length"] = doc.get("text_length", 0)
            doc["format"] = doc.get("format") or "html"
            dead_domain_skipped += 1
            update_progress(i + 1)
            continue

        # Skip if already downloaded
        if raw_file.exists() and raw_file.stat().st_size > 100:
            if not show_progress:
                logger.info(f"[{i+1}/{len(documents)}] Skipping {doc_id} (already exists)")
            with open(raw_file, "r", encoding="utf-8") as f:
                text = f.read()
            doc["text_length"] = len(text)
            doc["format"] = doc.get("format") or "html"
            if not doc.get("year"):
                valid_range = _parse_pope_dates(doc.get("pope_dates", ""))
                doc["year"] = _extract_year_from_text(text, valid_range=valid_range)
            if not doc.get("document_type") or doc.get("document_type") in {"document", "index"}:
                doc["document_type"] = infer_document_type(doc.get("title", ""), doc.get("url", ""), text)
            skipped += 1
            update_progress(i + 1)
            continue

        if not show_progress:
            logger.info(f"[{i+1}/{len(documents)}] Scraping: {doc['title'][:60]}...")

        result = extract_document_text(doc["url"], pope_dates=doc.get("pope_dates", ""))

        doc["format"] = result["format"]
        doc["language"] = result["language"]
        doc["text_length"] = len(result["text"])
        if result.get("year") and not doc.get("year"):
            doc["year"] = result["year"]
        if result.get("document_type"):
            doc["document_type"] = result["document_type"]

        if result["error"]:
            doc["error"] = result["error"]
            errors += 1
            if not show_progress:
                logger.warning(f"  Error: {result['error']}")
        else:
            extracted += 1
            if not show_progress:
                logger.info(f"  Extracted {len(result['text'])} chars ({result['language']})")

        # Save raw text
        if result["text"]:
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

        update_progress(i + 1)

    if show_progress:
        print()

    # Filter to English if requested
    if english_only:
        documents = [d for d in documents if d.get("language") == "en"]
        logger.info(f"Filtered to {len(documents)} English documents")

    return documents


def save_index(documents: list[dict]):
    """Save the document index to JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved index with {len(documents)} documents to {INDEX_FILE}")


def save_library_csv(documents: list[dict]):
    """Save LIBRARY.csv metadata table."""
    LIBRARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Backfill normalized author/category fields for older index rows.
    for doc in documents:
        pope_name = (doc.get("pope") or "").strip()
        is_council = (doc.get("category") == "council") or (pope_name.lower() == "church councils")
        doc["category"] = "council" if is_council else "pope"
        doc["author"] = doc.get("author") or (doc.get("title", "") if is_council else pope_name)
        doc["author_dates"] = doc.get("author_dates") or ("" if is_council else doc.get("pope_dates", ""))

    fieldnames = [
        "doc_id", "category", "author", "author_dates", "title", "year", "url",
        "document_type", "language", "format", "text_length",
        "pope", "pope_dates"
    ]

    with open(LIBRARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for doc in documents:
            writer.writerow(doc)

    logger.info(f"Saved LIBRARY.csv with {len(documents)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape Papal Encyclicals")
    parser.add_argument("--index-only", action="store_true",
                        help="Only scrape the directory index, not document texts")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Maximum number of documents to scrape (0=all)")
    parser.add_argument("--english-only", action="store_true",
                        help="Only keep English-language documents")
    args = parser.parse_args()

    # Step 1: Get document index
    if INDEX_FILE.exists():
        logger.info("Loading existing index...")
        with open(INDEX_FILE) as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from index")
    else:
        documents = scrape_index()
        save_index(documents)

    if args.index_only:
        logger.info("Index-only mode. Done.")
        return

    # Step 2: Scrape document texts
    documents = scrape_documents(
        documents,
        max_docs=args.max_docs,
        english_only=args.english_only,
    )

    # Step 3: Save updated index and LIBRARY.csv
    save_index(documents)
    save_library_csv(documents)

    # Summary
    total = len(documents)
    with_text = sum(1 for d in documents if d.get("text_length", 0) > 100)
    logger.info(f"\nScraping complete: {with_text}/{total} documents have text")


if __name__ == "__main__":
    main()

