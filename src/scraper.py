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
import logging
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.papalencyclicals.net"
DIRECTORY_URL = f"{BASE_URL}/document-directory"

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Parse the document directory
# ---------------------------------------------------------------------------

def fetch_page(url: str, retries: int = 3) -> str | None:
    """Fetch a page with retries and polite delay."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(2 ** (attempt + 1))
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None


def parse_directory(html: str) -> list[dict]:
    """
    Parse the document directory page.

    The page organizes documents under pope headings. Each pope section
    contains links to individual encyclical pages. We extract:
      - pope name
      - document title
      - document URL
      - year (parsed from title or context when available)
    """
    soup = BeautifulSoup(html, "lxml")
    documents = []

    # The directory page uses heading tags for pope names and lists/paragraphs
    # for document links. We look for patterns in the page structure.
    content = soup.find("div", class_="entry-content") or soup.find("article") or soup.find("main") or soup.body

    if content is None:
        logger.error("Could not find main content area on directory page")
        return documents

    current_pope = "Unknown"
    current_pope_dates = ""

    # Walk through all elements in the content area
    for element in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", "div", "strong", "b"]):
        tag = element.name

        # Pope headings are typically h2 or h3 tags, or bold/strong text
        if tag in ("h1", "h2", "h3", "h4"):
            text = element.get_text(strip=True)
            # Check if this looks like a pope name (not a generic page heading)
            if text and not any(skip in text.lower() for skip in [
                "document directory", "home", "menu", "search", "navigation",
                "footer", "sidebar", "comment", "share"
            ]):
                pope_match = re.match(
                    r"^((?:Pope\s+)?[\w\s.']+(?:I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI)?)\s*"
                    r"[\(\[]?\s*(\d{4}\s*[-–]\s*\d{4})?\s*[\)\]]?",
                    text.strip()
                )
                if pope_match:
                    current_pope = pope_match.group(1).strip()
                    current_pope_dates = pope_match.group(2) or ""
                else:
                    # Still use as pope name if it seems reasonable
                    if len(text) < 80 and not text.startswith("http"):
                        current_pope = text.strip()

        # Look for document links
        links = []
        if tag == "a":
            links = [element]
        else:
            links = element.find_all("a", href=True)

        for link in links:
            href = link.get("href", "")
            title = link.get_text(strip=True)

            if not href or not title:
                continue

            # Skip navigation, social media, and non-document links
            if any(skip in href.lower() for skip in [
                "facebook", "twitter", "instagram", "youtube",
                "#", "javascript:", "mailto:", "wp-content/uploads",
                "amazon", "donate", "about", "contact", "privacy",
                "comment", "reply", "login", "register"
            ]):
                continue

            # Build absolute URL
            if not href.startswith("http"):
                href = urljoin(BASE_URL, href)

            # Only include links that point to the same domain
            parsed = urlparse(href)
            if "papalencyclicals.net" not in parsed.netloc:
                continue

            # Skip if the link is to the directory itself
            if href.rstrip("/") == DIRECTORY_URL.rstrip("/"):
                continue

            # Try to extract year from title or surrounding text
            year = extract_year(title)

            # Avoid duplicate entries
            doc_id = make_doc_id(current_pope, title)

            documents.append({
                "doc_id": doc_id,
                "pope": current_pope,
                "pope_dates": current_pope_dates,
                "title": title,
                "url": href,
                "year": year,
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

    logger.info(f"Parsed {len(unique_docs)} unique documents from directory")
    return unique_docs


def extract_year(text: str) -> str:
    """Try to extract a 4-digit year from text."""
    match = re.search(r"\b(1[2-9]\d{2}|20[0-2]\d)\b", text)
    return match.group(1) if match else ""


def make_doc_id(pope: str, title: str) -> str:
    """Create a clean document ID from pope and title."""
    pope_clean = re.sub(r"[^\w]", "_", pope.strip())[:30]
    title_clean = re.sub(r"[^\w]", "_", title.strip())[:50]
    return f"{pope_clean}__{title_clean}".lower()


# ---------------------------------------------------------------------------
# Step 2: Extract text from individual document pages
# ---------------------------------------------------------------------------

def extract_document_text(url: str) -> dict:
    """
    Fetch and extract text from an individual encyclical page.

    Returns dict with keys: text, format, language, error
    """
    html = fetch_page(url)
    if html is None:
        return {"text": "", "format": "error", "language": "", "error": "fetch_failed"}

    soup = BeautifulSoup(html, "lxml")

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

    return {
        "text": text.strip(),
        "format": fmt,
        "language": language,
        "error": "" if text.strip() else "no_text_extracted",
    }


def extract_text_from_html(soup: BeautifulSoup) -> str:
    """Extract the main document text from an HTML page."""
    # Remove script, style, nav, header, footer elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer",
                               "aside", "form", "iframe"]):
        tag.decompose()

    # Try common content containers
    content = (
        soup.find("div", class_="entry-content")
        or soup.find("article")
        or soup.find("div", class_="post-content")
        or soup.find("div", class_="content")
        or soup.find("div", id="content")
        or soup.find("main")
    )

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
    try:
        import ebooklib
        from ebooklib import epub
        import tempfile

        resp = requests.get(epub_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
            f.write(resp.content)
            temp_path = f.name

        try:
            book = epub.read_epub(temp_path)
            texts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), "lxml")
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
                     english_only: bool = False) -> list[dict]:
    """
    Scrape text for each document in the index.
    Saves raw text files to data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = len(documents)
    if max_docs > 0:
        documents = documents[:max_docs]
        logger.info(f"Limiting to {max_docs} of {total} documents")

    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        raw_file = RAW_DIR / f"{doc_id}.txt"

        # Skip if already downloaded
        if raw_file.exists() and raw_file.stat().st_size > 100:
            logger.info(f"[{i+1}/{len(documents)}] Skipping {doc_id} (already exists)")
            with open(raw_file, "r", encoding="utf-8") as f:
                text = f.read()
            doc["text_length"] = len(text)
            doc["format"] = doc.get("format") or "html"
            continue

        logger.info(f"[{i+1}/{len(documents)}] Scraping: {doc['title'][:60]}...")

        result = extract_document_text(doc["url"])

        doc["format"] = result["format"]
        doc["language"] = result["language"]
        doc["text_length"] = len(result["text"])

        if result["error"]:
            doc["error"] = result["error"]
            logger.warning(f"  Error: {result['error']}")
        else:
            logger.info(f"  Extracted {len(result['text'])} chars ({result['language']})")

        # Save raw text
        if result["text"]:
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

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

    fieldnames = [
        "doc_id", "pope", "pope_dates", "title", "year", "url",
        "language", "format", "text_length"
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
