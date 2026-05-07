"""
Papal Documents Web Scraper
DS 5001 - Exploratory Text Analytics Final Project

Supports two sources:
    1) papalencyclicals.net directory corpus
    2) vatican.va pope section crawling (index pages -> document pages)

Usage:
        python src/scraper.py --source papalencyclicals
        python src/scraper.py --source vatican --pope benedict-xvi --lang en
        python src/scraper.py --source vatican --pope francis --index-only
"""

import os
import re
import csv
import json
import time
import warnings
import logging
import argparse
from collections import Counter
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
VATICAN_BASE_URL = "https://www.vatican.va"

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
DEAD_LINK_REPLACEMENTS_FILE = DATA_DIR / "processed" / "dead_link_replacements.csv"
POPE_COVERAGE_FILE = DATA_DIR / "processed" / "POPE_COVERAGE.csv"

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


def load_dead_link_replacements(csv_path: Path | None = None) -> dict[str, str]:
    """Load non-empty dead-link replacements keyed by doc_id."""
    path = csv_path or DEAD_LINK_REPLACEMENTS_FILE
    if not path.exists():
        logger.info(f"No dead-link replacement file found at {path}")
        return {}

    replacements = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            replacement_url = (row.get("replacement_url") or "").strip()
            if doc_id and replacement_url:
                replacements[doc_id] = replacement_url

    logger.info(f"Loaded {len(replacements)} dead-link replacements from {path}")
    return replacements


def apply_dead_link_replacements(
    documents: list[dict],
    replacements: dict[str, str] | None = None,
) -> tuple[list[dict], set[str]]:
    """Apply curated replacement URLs to document metadata in place."""
    replacement_map = replacements if replacements is not None else load_dead_link_replacements()
    replaced_doc_ids: set[str] = set()

    for doc in documents:
        doc_id = (doc.get("doc_id") or "").strip()
        replacement_url = replacement_map.get(doc_id, "")
        current_url = (doc.get("url") or "").strip()
        if not replacement_url or replacement_url == current_url:
            continue

        if current_url and not doc.get("original_url"):
            doc["original_url"] = current_url
        doc["url"] = replacement_url
        replaced_doc_ids.add(doc_id)

    if replaced_doc_ids:
        logger.info(f"Applied curated replacement URLs to {len(replaced_doc_ids)} documents")

    return documents, replaced_doc_ids


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
            "source": "papalencyclicals.net",
            "collection": "legacy_encyclicals",
            "detail_level": "standard",
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


def _normalize_url(url: str) -> str:
    """Normalize URL for de-duplication during crawling."""
    parsed = urlparse(url)
    clean_path = re.sub(r"/{2,}", "/", parsed.path)
    return f"{parsed.scheme}://{parsed.netloc}{clean_path}".rstrip("/")


def _build_vatican_scope_prefixes(
    pope_slug: str,
    lang: str,
    start_url: str = "",
    include_canonical: bool = True,
) -> list[str]:
    """Build allowed Vatican path prefixes for one pope/language crawl."""
    prefixes = []
    if include_canonical:
        prefixes.append(f"/content/{pope_slug.lower()}/{lang.lower()}")

    if start_url:
        parsed = urlparse(start_url)
        start_path = re.sub(r"/{2,}", "/", (parsed.path or "").rstrip("/")).lower()
        if start_path.endswith(".html") or start_path.endswith(".htm"):
            start_path = start_path.rsplit(".", 1)[0]
        if start_path:
            prefixes.append(start_path)

    deduped = []
    seen = set()
    for p in prefixes:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def _is_vatican_same_scope(url: str, scope_prefixes: list[str]) -> bool:
    """Keep crawl bounded to allowed Vatican path prefixes."""
    parsed = urlparse(url)
    if parsed.netloc.lower() != "www.vatican.va":
        return False

    path = parsed.path.lower()
    return any(path.startswith(prefix) for prefix in scope_prefixes)


def _is_vatican_document_url(url: str, scope_prefixes: list[str], lang: str) -> bool:
    """Heuristic for Vatican leaf documents."""
    if not _is_vatican_same_scope(url, scope_prefixes):
        return False

    path = urlparse(url).path.lower()
    filename = path.rsplit("/", 1)[-1]
    if not (filename.endswith(".html") or filename.endswith(".htm")):
        return False
    if "index" in filename:
        return False
    if path.endswith(f"/{lang.lower()}.html"):
        return False
    if "/_jcr_content/" in path:
        return False

    if "/documents/" in path:
        return True

    return bool(re.match(r"^(?:hf_|ben|fra|jpii|pio|let|msg|spe|hom)[\w\-\.]*\.(?:html|htm)$", filename))


def _extract_vatican_title(anchor) -> str:
    """Extract item title from link context (works when anchor text is just EN/IT/etc)."""
    raw = ""
    container = anchor.find_parent(["li", "p", "td", "tr", "div"])
    if container is not None:
        raw = container.get_text(" ", strip=True)
    if not raw:
        raw = anchor.get_text(" ", strip=True)

    raw = re.sub(r"\b([A-Z]{2})(\s*-\s*[A-Z]{2})+\b", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" -\u2013\u2014")
    if len(raw) < 4:
        raw = (anchor.get("title") or "").strip() or anchor.get_text(" ", strip=True)

    return raw or "Untitled"


def _vatican_doc_type_from_url(url: str, lang: str) -> str:
    """Infer Vatican document category from path segment after language."""
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    try:
        lang_idx = parts.index(lang)
    except ValueError:
        return "document"

    if lang_idx + 1 >= len(parts):
        return "document"

    seg = parts[lang_idx + 1].lower()
    mapping = {
        "angelus": "angelus",
        "apost_constitutions": "apostolic constitution",
        "apost_exhortations": "apostolic exhortation",
        "apost_letters": "apostolic letter",
        "audiences": "audience",
        "biography": "biography",
        "encyclicals": "encyclical",
        "homilies": "homily",
        "letters": "letter",
        "messages": "message",
        "motu_proprio": "motu proprio",
        "prayers": "prayer",
        "speeches": "speech",
        "documentation": "documentation",
        "travels": "travel",
    }
    return mapping.get(seg, seg.replace("_", " "))


def _format_pope_display_name(pope_slug: str) -> str:
    """Convert pope slug to display label while preserving Roman numerals."""
    parts = []
    for token in pope_slug.split("-"):
        if re.fullmatch(r"[ivxlcdm]+", token):
            parts.append(token.upper())
        else:
            parts.append(token.capitalize())
    return f"Pope {' '.join(parts)}"


def _vatican_doc_id(url: str, pope_slug: str, lang: str, scope_prefixes: list[str] | None = None) -> str:
    """Generate stable doc_id for Vatican documents."""
    path = urlparse(url).path.lower()
    tail = path.strip("/")
    if scope_prefixes:
        for scope_prefix in sorted(scope_prefixes, key=len, reverse=True):
            scope = f"{scope_prefix.rstrip('/')}/"
            if path.startswith(scope):
                tail = path[len(scope):]
                break
    if tail == path.strip("/"):
        scope = f"/content/{pope_slug.lower()}/{lang.lower()}/"
        if scope in path:
            tail = path.split(scope, 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")[:120]
    return f"vatican__{pope_slug.lower()}__{lang.lower()}__{slug}"


def discover_vatican_pope_links(source_url: str = "https://www.vatican.va/content/vatican/en.html") -> list[dict]:
    """Discover pope profile card links from the Vatican English home page."""
    html = fetch_page(source_url)
    if html is None:
        logger.warning(f"Failed to fetch Vatican source page: {source_url}")
        return []

    soup = _make_soup(html)
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(source_url, a.get("href", "").strip())
        path = urlparse(href).path.lower()
        if "/content/vatican/en/holy-father/" not in path:
            continue
        if not path.endswith(".html"):
            continue
        normalized = _normalize_url(href)
        if normalized in seen:
            continue
        seen.add(normalized)
        title = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        links.append({
            "pope_label": title,
            "pope_card_url": normalized,
        })

    logger.info(f"Discovered {len(links)} pope profile links from {source_url}")
    return links


def _resolve_vatican_pope_slug_from_card(card_url: str, lang: str = "en") -> tuple[str, str]:
    """Resolve canonical pope slug/home URL from a Vatican pope profile card page."""
    html = fetch_page(card_url, retries=2)
    if html is None:
        return "", ""

    soup = _make_soup(html)
    pattern = re.compile(rf"/content/([^/]+)/{re.escape(lang)}/?\.html$", re.IGNORECASE)
    candidates: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(card_url, a.get("href", "").strip())
        path = urlparse(href).path
        match = pattern.search(path)
        if not match:
            continue
        slug = match.group(1).lower()
        if slug in {"vatican", "romancuria", "liturgy", "photogallery"}:
            continue

        normalized_href = _normalize_url(href)
        if slug not in candidates:
            candidates[slug] = normalized_href

    if not candidates:
        return "", ""

    card_slug = urlparse(card_url).path.rstrip("/").rsplit("/", 1)[-1].replace(".html", "").lower()
    candidate_list = list(candidates.keys())
    selected_slug = _select_best_slug_for_card(card_slug, candidate_list)

    # Some profile pages do not expose their own /content/<slug>/<lang>.html
    # link in static HTML. In that case, build a best-effort translated slug
    # from the profile card path.
    translated_slug = _translate_card_slug(card_slug)
    if selected_slug and _slug_match_score(card_slug, selected_slug) == (0, 0, 0):
        if translated_slug and translated_slug not in {"", "vatican"}:
            return translated_slug, f"https://www.vatican.va/content/{translated_slug}/{lang}.html"

    return selected_slug, candidates[selected_slug]


def _select_best_slug_for_card(card_slug: str, candidate_slugs: list[str]) -> str:
    """Pick the most likely pope content slug from a profile card slug."""
    if not candidate_slugs:
        return ""

    if card_slug in candidate_slugs:
        return card_slug

    translated_slug = _translate_card_slug(card_slug)
    if translated_slug in candidate_slugs:
        return translated_slug

    ranked = sorted(candidate_slugs, key=lambda c: _slug_match_score(card_slug, c), reverse=True)
    return ranked[0]


def _translate_card_slug(card_slug: str) -> str:
    """Translate common Italian pope-name tokens to English equivalents."""
    name_map = {
        "francesco": "francis",
        "benedetto": "benedict",
        "giovanni": "john",
        "paolo": "paul",
        "pio": "pius",
        "leone": "leo",
        "clemente": "clement",
        "gregorio": "gregory",
        "innocenzo": "innocent",
        "alessandro": "alexander",
        "adriano": "adrian",
        "niccolo": "nicholas",
        "martino": "martin",
        "urbano": "urban",
        "eugenio": "eugene",
        "onorio": "honorius",
        "celestino": "celestine",
        "stefano": "stephen",
        "sisto": "sixtus",
        "zaccaria": "zachary",
        "silvestro": "sylvester",
        "teodoro": "theodore",
        "bonifacio": "boniface",
        "anastasio": "anastasius",
        "costantino": "constantine",
        "sergio": "sergius",
        "vigilio": "vigilius",
        "pelagio": "pelagius",
        "ormisda": "hormisdas",
        "landone": "lando",
        "damaso": "damasus",
        "marino": "marinus",
        "agapito": "agapetus",
        "callisto": "callixtus",
        "giulio": "julius",
        "lucio": "lucius",
    }

    parts = card_slug.split("-")
    translated_parts = [name_map.get(p, p) for p in parts]
    return "-".join(translated_parts)


def _slug_match_score(card_slug: str, candidate: str) -> tuple[int, int, int]:
    """Score candidate slug similarity to profile card slug."""
    parts = card_slug.split("-")
    translated_parts = _translate_card_slug(card_slug).split("-")

    card_parts = set(parts)
    translated_set = set(translated_parts)

    cand_parts = set(candidate.split("-"))
    overlap_raw = len(card_parts.intersection(cand_parts))
    overlap_translated = len(translated_set.intersection(cand_parts))
    roman_bonus = 1 if any(re.fullmatch(r"[ivxlcdm]+", p) and p in cand_parts for p in parts) else 0
    return (overlap_translated, overlap_raw, roman_bonus)


def discover_vatican_pope_destinations(
    source_url: str = "https://www.vatican.va/content/vatican/en.html",
    lang: str = "en",
    max_popes: int = 0,
) -> list[dict]:
    """Discover pope profile links and resolve crawl-ready Vatican slugs/home URLs."""
    profile_links = discover_vatican_pope_links(source_url=source_url)
    if max_popes > 0:
        profile_links = profile_links[:max_popes]

    rows = []
    for row in profile_links:
        slug, home_url = _resolve_vatican_pope_slug_from_card(row["pope_card_url"], lang=lang)
        out = {
            "pope_label": row["pope_label"],
            "pope_card_url": row["pope_card_url"],
            "pope_slug": slug,
            "pope_home_url": home_url,
        }
        rows.append(out)

    resolved = sum(1 for r in rows if r["pope_slug"])
    logger.info(f"Resolved crawl slugs for {resolved}/{len(rows)} pope profiles")
    return rows


def scrape_vatican_pope_index(
    pope_slug: str,
    lang: str = "en",
    start_url: str = "",
    max_index_pages: int = 500,
) -> list[dict]:
    """Crawl one Vatican pope section and collect leaf document URLs."""
    canonical_start = f"{VATICAN_BASE_URL}/content/{pope_slug}/{lang}.html"
    explicit_start = bool(start_url.strip())
    effective_start_url = _normalize_url(start_url.strip()) if explicit_start else canonical_start
    scope_prefixes = _build_vatican_scope_prefixes(
        pope_slug=pope_slug,
        lang=lang,
        start_url=effective_start_url,
        include_canonical=not explicit_start,
    )

    queue = [effective_start_url]
    if (not explicit_start) and (effective_start_url != canonical_start):
        queue.append(canonical_start)
    visited = set()
    documents_by_url: dict[str, dict] = {}

    logger.info(f"Crawling Vatican index pages from {effective_start_url}")
    pope_display = _format_pope_display_name(pope_slug)

    while queue and len(visited) < max_index_pages:
        current = queue.pop(0)
        norm_current = _normalize_url(current)
        if norm_current in visited:
            continue
        visited.add(norm_current)

        html = fetch_page(current)
        if html is None:
            continue

        soup = _make_soup(html)
        for anchor in soup.find_all("a", href=True):
            href = _normalize_url(urljoin(current, anchor.get("href", "").strip()))
            if not href or not _is_vatican_same_scope(href, scope_prefixes):
                continue

            if _is_vatican_document_url(href, scope_prefixes, lang):
                if href not in documents_by_url:
                    title = _extract_vatican_title(anchor)
                    documents_by_url[href] = {
                        "doc_id": _vatican_doc_id(href, pope_slug, lang, scope_prefixes=scope_prefixes),
                        "category": "pope",
                        "author": pope_display,
                        "author_dates": "",
                        "source": "vatican.va",
                        "collection": "modern_popes_fulltext",
                        "detail_level": "detailed",
                        "pope": pope_display,
                        "pope_dates": "",
                        "title": title,
                        "url": href,
                        "year": extract_year(title),
                        "document_type": _vatican_doc_type_from_url(href, lang),
                        "language": lang.lower(),
                        "format": "",
                    }
                continue

            path = urlparse(href).path.lower()
            filename = path.rsplit("/", 1)[-1]
            # Queue remaining HTML navigation pages (includes year pages).
            if (filename.endswith(".html") or filename.endswith(".htm")) and href not in visited and href not in queue:
                queue.append(href)

    logger.info(
        f"Vatican crawl complete for {pope_slug}/{lang}: "
        f"visited={len(visited)} index pages, documents={len(documents_by_url)}"
    )
    return list(documents_by_url.values())


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
_ISSUE_YEAR_RE = re.compile(
    r"\b(?:issued|promulgated|published|dated|enacted|proclaimed|approved|signed)\b"
    r"[^\d]{0,50}\b(?:in\s+)?(1[2-9]\d{2}|20[0-2]\d)\b",
    re.IGNORECASE,
)


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

    # Pass -1: explicit issuance phrases often appear in title/subtitle blocks.
    header_parts = []
    if soup.title and soup.title.get_text(" ", strip=True):
        header_parts.append(soup.title.get_text(" ", strip=True))
    for tag_name in ("h1", "h2", "h3", "h4"):
        for tag in soup.find_all(tag_name)[:3]:
            text = tag.get_text(" ", strip=True)
            if text:
                header_parts.append(text)
    for meta_name in ("description", "og:description", "twitter:description"):
        meta = soup.find("meta", attrs={"name": meta_name}) or soup.find("meta", attrs={"property": meta_name})
        if meta:
            value = (meta.get("content") or "").strip()
            if value:
                header_parts.append(value)

    header_blob = " ".join(header_parts)
    m_issue = _ISSUE_YEAR_RE.search(header_blob)
    if m_issue and _plausible(m_issue.group(1)):
        return m_issue.group(1)

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
    m_issue = _ISSUE_YEAR_RE.search(full_text[:2500])
    if m_issue and _plausible(m_issue.group(1)):
        return m_issue.group(1)
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

    # Pass 0: explicit issuance phrase near front matter.
    m_issue = _ISSUE_YEAR_RE.search(text[:2500])
    if m_issue and _plausible(m_issue.group(1)):
        return m_issue.group(1)

    # Pass 1: search the last 1000 chars first — subscription date is always
    # at the end of the document.
    tail = text[-1000:] if len(text) > 1000 else text
    for m in _DATE_RE.finditer(tail):
        if _plausible(m.group(1)):
            return m.group(1)

    # Pass 2: month + year, full document
    for m in _DATE_RE.finditer(text):
        if _plausible(m.group(1)):
            return m.group(1)

    # Pass 3: standalone year within valid range
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

    # Structured extraction: gather text from semantic block elements.
    # Include <center> because some older pages use it for sub-headings and
    # issuance lines (e.g., papalencyclicals.net older constitutions).
    paragraphs = []
    for p in content.find_all(["p", "blockquote", "center",
                                "h1", "h2", "h3", "h4", "h5", "h6"]):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    para_text = "\n\n".join(paragraphs)

    # Full-content fallback via get_text(), which also captures raw text nodes
    # that live outside block elements.  Some older pages place the entire body
    # text in bare text nodes rather than in <p> tags; in those cases the
    # paragraph extraction above only returns a small fragment of the real text.
    full_text = re.sub(r"\n{3,}", "\n\n", content.get_text(separator="\n", strip=True))

    # Use structured extraction when it captures most of the content.
    # Fall back to get_text() when significant text is in raw text nodes.
    if para_text.strip() and len(para_text.strip()) >= 0.6 * len(full_text.strip()):
        return para_text
    return full_text


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
    """Language detection using stopword frequency over a text sample."""
    if not text:
        return "unknown"

    sample = text[:12000].lower()
    tokens = re.findall(r"[a-z\u00c0-\u00ff]+", sample)
    if len(tokens) < 40:
        return "unknown"

    stopwords = {
        "en": {
            "the", "and", "of", "to", "in", "that", "for", "with", "is", "on", "as", "by",
            "be", "this", "we", "our", "are", "from", "or", "which", "an", "at", "it", "not",
        },
        "la": {
            "et", "in", "est", "non", "qui", "quae", "quod", "sed", "cum", "ad", "ut", "sunt",
            "autem", "hoc", "haec", "ecclesia", "christi", "dei", "romana", "apostolica", "sanctae",
            "nostra", "fidei", "vitae", "quam", "omnibus", "ipsa", "eius", "per", "pro", "ita",
        },
        "it": {
            "il", "la", "lo", "gli", "le", "dei", "degli", "della", "delle", "del", "di", "che",
            "nella", "nelle", "alla", "alle", "con", "per", "non", "come", "sono", "questo", "questa",
            "quello", "quella", "anche", "nel", "una", "un", "si", "dei", "dai", "agli",
        },
        "fr": {
            "le", "la", "les", "des", "de", "du", "dans", "qui", "est", "pour", "avec", "nous",
            "vous", "sur", "par", "une", "un", "et", "ce", "cette", "aux", "au", "pas", "que",
            "plus", "comme", "ont", "ils", "elles", "leur", "leurs",
        },
    }

    token_counts = Counter(tokens)
    token_set = set(token_counts)

    scores = {}
    for lang, words in stopwords.items():
        hit_count = sum(token_counts[w] for w in words if w in token_counts)
        unique_hits = len(token_set.intersection(words))
        freq_component = hit_count / len(tokens)
        coverage_component = unique_hits / len(words)
        scores[lang] = (0.75 * freq_component) + (0.25 * coverage_component)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_lang, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < 0.015:
        return "unknown"
    if second_score > 0 and best_score < (second_score * 1.15):
        return "unknown"
    return best_lang


def recheck_languages_from_raw(
    documents: list[dict],
    source: str = "papalencyclicals.net",
    only_if_current: str | None = "en",
    min_text_chars: int = 500,
) -> dict:
    """Reclassify document language from local raw files only (no network calls)."""
    updated = []
    skipped_missing_raw = 0
    skipped_short = 0
    skipped_filter = 0
    examined = 0

    source_norm = source.strip().lower()
    only_if_current_norm = (only_if_current or "").strip().lower()

    for doc in documents:
        doc_source = (doc.get("source") or "").strip().lower()
        if source_norm and doc_source != source_norm:
            skipped_filter += 1
            continue

        current_lang = (doc.get("language") or "unknown").strip().lower()
        if only_if_current_norm and current_lang != only_if_current_norm:
            skipped_filter += 1
            continue

        doc_id = doc.get("doc_id")
        if not doc_id:
            skipped_filter += 1
            continue

        raw_file = RAW_DIR / f"{doc_id}.txt"
        if not raw_file.exists() or raw_file.stat().st_size <= 100:
            skipped_missing_raw += 1
            continue

        text = raw_file.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < min_text_chars:
            skipped_short += 1
            continue

        examined += 1
        new_lang = detect_language_simple(text)
        if new_lang and new_lang != "unknown" and new_lang != current_lang:
            doc["language"] = new_lang
            updated.append(
                {
                    "doc_id": doc_id,
                    "old_language": current_lang,
                    "new_language": new_lang,
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                }
            )

    return {
        "total_documents": len(documents),
        "examined": examined,
        "updated": len(updated),
        "skipped_missing_raw": skipped_missing_raw,
        "skipped_short": skipped_short,
        "skipped_filter": skipped_filter,
        "changes": updated,
    }


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

    documents, replaced_doc_ids = apply_dead_link_replacements(documents)

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
        has_replacement = doc_id in replaced_doc_ids

        if has_replacement:
            doc.pop("error", None)

        # Skip known dead domains to avoid repeated DNS/connection noise.
        if _domain(doc.get("url", "")) in KNOWN_DEAD_DOMAINS:
            doc["error"] = "known_dead_domain"
            doc["text_length"] = doc.get("text_length", 0)
            doc["format"] = doc.get("format") or "html"
            dead_domain_skipped += 1
            update_progress(i + 1)
            continue

        # Skip if already downloaded
        if raw_file.exists() and raw_file.stat().st_size > 100 and not has_replacement:
            if not show_progress:
                logger.info(f"[{i+1}/{len(documents)}] Skipping {doc_id} (already exists)")
            with open(raw_file, "r", encoding="utf-8") as f:
                text = f.read()
            doc["text_length"] = len(text)
            doc["format"] = doc.get("format") or "html"
            if not doc.get("year"):
                valid_range = _parse_pope_dates(doc.get("pope_dates", ""))
                doc["year"] = _extract_year_from_text(text, valid_range=valid_range)
                if not doc.get("year") and doc.get("url"):
                    page_html = fetch_page(doc["url"], retries=2)
                    if page_html is not None:
                        doc["year"] = _extract_year_from_page(
                            _make_soup(page_html),
                            valid_range=valid_range,
                        )
            if not doc.get("document_type") or doc.get("document_type") in {"document", "index"}:
                doc["document_type"] = infer_document_type(doc.get("title", ""), doc.get("url", ""), text)
            skipped += 1
            update_progress(i + 1)
            continue

        if not show_progress:
            logger.info(f"[{i+1}/{len(documents)}] Scraping: {doc['title'][:60]}...")

        result = extract_document_text(doc["url"], pope_dates=doc.get("pope_dates", ""))

        doc["format"] = result["format"]
        # Don't downgrade a previously-valid language to "unknown" when the
        # freshly scraped page yielded very little text (e.g., tiny index pages
        # that are re-fetched every run because their raw file is < 100 bytes).
        new_lang = result["language"]
        if new_lang and new_lang != "unknown":
            doc["language"] = new_lang
        elif not doc.get("language") or doc["language"] == "unknown":
            doc["language"] = new_lang
        # else: keep existing valid language from a prior scrape run
        doc["text_length"] = len(result["text"])
        if result.get("year") and not doc.get("year"):
            doc["year"] = result["year"]
        if result.get("document_type"):
            keep_existing_vatican_type = (
                doc.get("source") == "vatican.va"
                and (doc.get("document_type") or "").strip().lower() not in {"", "document", "index"}
            )
            if not keep_existing_vatican_type:
                doc["document_type"] = result["document_type"]

        if result["error"]:
            doc["error"] = result["error"]
            errors += 1
            if not show_progress:
                logger.warning(f"  Error: {result['error']}")
        else:
            doc.pop("error", None)
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


def load_index() -> list[dict]:
    """Load the existing index if present."""
    if not INDEX_FILE.exists():
        return []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_documents(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Merge incoming document records into existing index using doc_id as key."""
    merged = {doc.get("doc_id"): doc for doc in existing if doc.get("doc_id")}
    for doc in incoming:
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        if doc_id in merged:
            updated = merged[doc_id]
            updated.update(doc)
            merged[doc_id] = updated
        else:
            merged[doc_id] = doc
    return list(merged.values())


def save_pope_coverage_csv(documents: list[dict]):
    """Save per-pope/source coverage metrics for corpus segmentation."""
    POPE_COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], dict] = {}
    for doc in documents:
        pope = (doc.get("pope") or "Unknown").strip() or "Unknown"
        source = (doc.get("source") or "unknown").strip() or "unknown"
        key = (pope, source)
        if key not in groups:
            groups[key] = {
                "pope": pope,
                "source": source,
                "detail_level": "standard",
                "n_documents": 0,
                "n_with_text": 0,
                "languages": set(),
            }

        row = groups[key]
        row["n_documents"] += 1
        if int(doc.get("text_length", 0) or 0) > 100:
            row["n_with_text"] += 1
        lang = (doc.get("language") or "").strip()
        if lang:
            row["languages"].add(lang)
        if (doc.get("detail_level") or "").strip().lower() == "detailed":
            row["detail_level"] = "detailed"

    fieldnames = [
        "pope", "source", "detail_level", "n_documents", "n_with_text", "languages"
    ]
    with open(POPE_COVERAGE_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            row = groups[key]
            writer.writerow({
                "pope": row["pope"],
                "source": row["source"],
                "detail_level": row["detail_level"],
                "n_documents": row["n_documents"],
                "n_with_text": row["n_with_text"],
                "languages": ";".join(sorted(row["languages"])),
            })

    logger.info(f"Saved pope coverage summary to {POPE_COVERAGE_FILE}")


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
        doc["source"] = doc.get("source") or "papalencyclicals.net"
        doc["collection"] = doc.get("collection") or "legacy_encyclicals"
        doc["detail_level"] = doc.get("detail_level") or "standard"

    fieldnames = [
        "doc_id", "category", "author", "author_dates", "title", "year", "url",
        "original_url",
        "source", "collection", "detail_level",
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
    parser = argparse.ArgumentParser(description="Scrape papal documents")
    parser.add_argument("--source", choices=["papalencyclicals", "vatican"], default="papalencyclicals",
                        help="Source corpus to scrape")
    parser.add_argument("--pope", default="",
                        help="Vatican pope slug (example: benedict-xvi, francis, john-paul-ii)")
    parser.add_argument("--lang", default="en",
                        help="Language code for Vatican crawl scope (default: en)")
    parser.add_argument("--max-index-pages", type=int, default=500,
                        help="Maximum Vatican index/navigation pages to crawl")
    parser.add_argument("--index-only", action="store_true",
                        help="Only scrape the directory index, not document texts")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Maximum number of documents to scrape (0=all)")
    parser.add_argument("--english-only", action="store_true",
                        help="Only keep English-language documents")
    parser.add_argument("--recheck-papal-languages", action="store_true",
                        help="Recheck papalencyclicals language labels from local raw files only")
    parser.add_argument("--recheck-only-current-en", action="store_true",
                        help="When rechecking, only evaluate rows currently tagged as English")
    args = parser.parse_args()

    existing_docs = load_index()
    if existing_docs:
        logger.info(f"Loaded {len(existing_docs)} existing documents from index")

    if args.recheck_papal_languages:
        if not existing_docs:
            logger.info("No existing index found; nothing to recheck.")
            return

        report = recheck_languages_from_raw(
            existing_docs,
            source="papalencyclicals.net",
            only_if_current="en" if args.recheck_only_current_en else None,
        )
        save_index(existing_docs)
        save_library_csv(existing_docs)
        save_pope_coverage_csv(existing_docs)
        logger.info(
            "Language recheck complete: "
            f"examined={report['examined']} updated={report['updated']} "
            f"missing_raw={report['skipped_missing_raw']} short={report['skipped_short']}"
        )
        return

    if args.source == "papalencyclicals":
        if existing_docs:
            documents = existing_docs
        else:
            documents = scrape_index()
            save_index(documents)

        if args.index_only:
            logger.info("Index-only mode. Done.")
            return

        documents = scrape_documents(
            documents,
            max_docs=args.max_docs,
            english_only=args.english_only,
        )

        save_index(documents)
        save_library_csv(documents)
        save_pope_coverage_csv(documents)

        total = len(documents)
        with_text = sum(1 for d in documents if d.get("text_length", 0) > 100)
        logger.info(f"\nScraping complete: {with_text}/{total} documents have text")
        return

    # Vatican flow
    if not args.pope:
        raise ValueError("--pope is required when --source vatican")

    vatican_docs = scrape_vatican_pope_index(
        pope_slug=args.pope,
        lang=args.lang,
        max_index_pages=args.max_index_pages,
    )

    if args.index_only:
        combined = merge_documents(existing_docs, vatican_docs)
        save_index(combined)
        save_library_csv(combined)
        save_pope_coverage_csv(combined)
        logger.info("Index-only mode. Vatican index merged into corpus.")
        return

    scraped_vatican_docs = scrape_documents(
        vatican_docs,
        max_docs=args.max_docs,
        english_only=args.english_only,
        show_progress=True,
        progress_label=f"Vatican {args.pope}/{args.lang}",
    )

    combined = merge_documents(existing_docs, scraped_vatican_docs)
    save_index(combined)
    save_library_csv(combined)
    save_pope_coverage_csv(combined)

    total_new = len(scraped_vatican_docs)
    with_text_new = sum(1 for d in scraped_vatican_docs if d.get("text_length", 0) > 100)
    logger.info(f"\nVatican scrape complete: {with_text_new}/{total_new} new documents have text")
    logger.info(f"Combined corpus size: {len(combined)} documents")


if __name__ == "__main__":
    main()

