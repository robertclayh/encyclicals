"""
Build index entries for existing Vatican raw text files.

Vatican raw files follow the naming pattern:
  vatican__POPE_SLUG__LANG__DESCRIPTION.txt

This script:
1. Scans data/raw for vatican__*.txt files
2. Parses the filename to extract doc_id, pope_slug, language
3. Reads first few lines to estimate document info
4. Merges new entries into encyclicals_index.json
5. Saves the updated index
"""

import json
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_FILE = DATA_DIR / "encyclicals_index.json"

def parse_vatican_filename(filename):
    """
    Parse vatican__POPE__LANG__DESCRIPTION.txt to extract components.
    Returns: (doc_id, pope_slug, lang, description) or None if not vatican file
    """
    if not filename.startswith("vatican__"):
        return None
    
    # Remove .txt extension
    base = filename[:-4] if filename.endswith(".txt") else filename
    
    # Split by __
    parts = base.split("__")
    if len(parts) < 3:
        return None
    
    # parts[0] = "vatican"
    # parts[1] = pope_slug
    # parts[2] = lang
    # parts[3+] = description (joined back with __)
    
    pope_slug = parts[1]
    lang = parts[2]
    description = "__".join(parts[3:]) if len(parts) > 3 else ""
    
    doc_id = filename[:-4]  # Remove .txt
    
    return doc_id, pope_slug, lang, description

def extract_text_info(filepath, description, max_chars=200):
    """
    Use file size and parse description to create placeholder title.
    Returns: (text_length, title_hint)
    """
    try:
        text_length = filepath.stat().st_size
        # Create title from description by replacing underscores with spaces
        title_hint = description.replace("_", " ").replace("-", " ").title()[:200] if description else f"Vatican Document"
        return text_length, title_hint
    except Exception as e:
        logger.error(f"Failed to stat {filepath}: {e}")
        return 0, ""

def build_vatican_index_entries():
    """
    Scan raw folder for vatican__ files and build index entries.
    """
    if not INDEX_FILE.exists():
        logger.error(f"Index file not found: {INDEX_FILE}")
        return None
    
    # Load existing index
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    existing_doc_ids = {doc['doc_id'] for doc in documents}
    logger.info(f"Loaded {len(documents)} documents from index")
    
    # Scan for Vatican files
    vatican_files = sorted(RAW_DIR.glob("vatican__*.txt"))
    logger.info(f"Found {len(vatican_files)} vatican__ files in raw folder")
    
    new_entries = 0
    skipped_entries = 0
    
    for idx, raw_file in enumerate(vatican_files):
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processing: {idx + 1} / {len(vatican_files)}")
        
        parsed = parse_vatican_filename(raw_file.name)
        if not parsed:
            logger.warning(f"Failed to parse filename: {raw_file.name}")
            skipped_entries += 1
            continue
        
        doc_id, pope_slug, lang, description = parsed
        
        # Skip if already in index
        if doc_id in existing_doc_ids:
            skipped_entries += 1
            continue
        
        # Extract text info
        text_length, title_hint = extract_text_info(raw_file, description, max_chars=200)
        
        # Build doc entry (matching the structure expected by pipeline)
        entry = {
            "doc_id": doc_id,
            "pope": pope_slug.replace("-", " ").title(),  # e.g., "benedict-xvi" -> "Benedict Xvi"
            "pope_slug": pope_slug,
            "title": title_hint or f"Document from {pope_slug}",
            "url": f"https://www.vatican.va/content/{pope_slug}/",  # Placeholder - would need actual URL
            "year": None,  # Would need to extract from filename or content
            "language": lang,
            "text_length": text_length,
            "category": "pope",
            "document_type": "papal_document",
            "source": "vatican.va",
            "format": "text"
        }
        
        documents.append(entry)
        new_entries += 1
    
    logger.info(f"\nSummary:")
    logger.info(f"  New entries created: {new_entries}")
    logger.info(f"  Already in index: {skipped_entries}")
    logger.info(f"  Total documents now: {len(documents)}")
    
    return documents, new_entries

def main():
    logger.info("Building Vatican index entries from raw files...")
    logger.info("")
    
    result = build_vatican_index_entries()
    if result is None:
        logger.error("Failed to build index")
        return
    
    documents, new_entries = result
    
    if new_entries > 0:
        logger.info(f"\nSaving {len(documents)} documents to index...")
        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Index saved successfully!")
        logger.info(f"✓ Added {new_entries} Vatican document entries")
    else:
        logger.info("No new entries to add.")

if __name__ == "__main__":
    main()
