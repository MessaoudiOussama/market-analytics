"""
ECB Historical Backfill Scraper
---------------------------------
Fetches historical ECB speeches using their internal JSON database (foedb).

The ECB stores all publications in a chunked JSON database at:
    /foedb/dbs/foedb/publications.en/{version}/{hash}/

Steps:
    1. Fetch the type index for speeches (type 18) to get all speech record IDs
    2. Fetch the data chunks containing those records
    3. For each speech, scrape the full text from the linked page
    4. Store in PostgreSQL

This allows backfilling hundreds of speeches at once.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

ECB_BASE_URL = "https://www.ecb.europa.eu"

# ECB foedb JSON database endpoints
FOEDB_BASE = "https://www.ecb.europa.eu/foedb/dbs/foedb/publications.en"
FOEDB_VERSIONS_URL = f"{FOEDB_BASE}/versions.json"

# Type 18 = Speeches in the ECB publications database
SPEECH_TYPE_ID = "18"

# How many records per chunk in the data files
CHUNK_SIZE = 250
GROUP_SIZE = 1000
# Number of fields per record in the flat data array
FIELDS_PER_RECORD = 13


@dataclass
class ECBArchiveSpeech:
    """Represents a speech from the ECB archive."""

    title: str
    speaker: str
    content: str
    published_at: datetime
    url: str
    source: str = "ecb"


def _fetch_json(url: str) -> Any:
    """Fetch and parse JSON from a URL, handling SSL issues."""
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch JSON from {url}: {e}")
        return None


def get_latest_version() -> tuple[str, str] | None:
    """
    Get the latest version/hash of the ECB publications database.

    Returns:
        Tuple of (timestamp, hash) or None if failed.
    """
    versions = _fetch_json(FOEDB_VERSIONS_URL)
    if not versions or not isinstance(versions, list) or len(versions) == 0:
        logger.error("Could not fetch foedb versions")
        return None

    latest = versions[0]
    # Format: {"version": "1770390012", "hash": "7Y2zqYoh"}
    return str(latest["version"]), latest["hash"]


def get_speech_record_ids(timestamp: str, hash: str) -> list[int]:
    """
    Get all record IDs that are speeches (type 18).

    Args:
        timestamp: Database version timestamp.
        hash: Database version hash.

    Returns:
        List of record IDs for speeches.
    """
    all_ids = []
    chunk_idx = 0

    while True:
        url = f"{FOEDB_BASE}/{timestamp}/{hash}/indexes/type/{SPEECH_TYPE_ID}/chunk_{chunk_idx}.json"
        data = _fetch_json(url)

        if data is None:
            break

        if isinstance(data, list):
            all_ids.extend(data)
            logger.info(f"Fetched speech index chunk {chunk_idx}: {len(data)} IDs")

            # If we got fewer IDs than expected, we've reached the last chunk
            if len(data) < 2500:  # Index chunk size
                break
        else:
            break

        chunk_idx += 1

    logger.info(f"Total speech record IDs: {len(all_ids)}")
    return all_ids


def _chunk_flat_array(flat_data: list, fields: int) -> list[list]:
    """
    Split a flat array into records of N fields each.

    The ECB foedb stores data as one flat array where every `fields` consecutive
    elements form one record. E.g., with 13 fields:
        [id1, ts1, year1, ..., meta1, id2, ts2, year2, ..., meta2, ...]

    Args:
        flat_data: The flat array from the chunk JSON.
        fields: Number of fields per record.

    Returns:
        List of record arrays, each with `fields` elements.
    """
    return [flat_data[i:i + fields] for i in range(0, len(flat_data), fields)]


def get_records_from_chunks(
    timestamp: str, hash_val: str, speech_indices: set[int], max_records: int = 500
) -> list[dict]:
    """
    Fetch publication records from data chunks.

    Data is stored as a FLAT array where every 13 consecutive elements form one record.
    250 records per chunk × 13 fields = 3250 elements per chunk JSON.

    The speech_indices are ROW POSITIONS (record numbers) in the global record sequence.
    Data is split into groups of 1000 records, each split into chunks of 250 records.

    Args:
        timestamp: Database version timestamp.
        hash_val: Database version hash.
        speech_indices: Set of global row indices for speeches.
        max_records: Maximum number of records to fetch.

    Returns:
        List of parsed record dicts.
    """
    records = []
    global_row_idx = 0  # Tracks record position across all chunks
    group_idx = 0
    chunk_in_group = 0

    while len(records) < max_records:
        url = f"{FOEDB_BASE}/{timestamp}/{hash_val}/data/{group_idx}/chunk_{chunk_in_group}.json"

        data = _fetch_json(url)
        if data is None:
            break

        if not isinstance(data, list):
            break

        # Split flat array into individual records
        chunk_records = _chunk_flat_array(data, FIELDS_PER_RECORD)

        for record in chunk_records:
            if global_row_idx in speech_indices:
                parsed = parse_record(record)
                if parsed:
                    records.append(parsed)
                    if len(records) >= max_records:
                        break

            global_row_idx += 1

        logger.info(
            f"Scanned data/{group_idx}/chunk_{chunk_in_group}: "
            f"{len(chunk_records)} records in chunk, "
            f"found {len(records)} speeches so far (row {global_row_idx})"
        )

        if len(records) >= max_records:
            break

        # Move to next chunk
        chunk_in_group += 1
        if chunk_in_group >= (GROUP_SIZE // CHUNK_SIZE):
            chunk_in_group = 0
            group_idx += 1

        # Safety: stop if we've scanned all records
        if global_row_idx > 20000:
            break

    return records


def parse_record(record: list) -> dict | None:
    """
    Parse a single foedb record (list of 13 fields) into a speech metadata dict.

    Field layout (based on metadata.json dataFields):
        [0]  id (int)
        [1]  pub_timestamp (int, unix)
        [2]  year (int)
        [3]  issue_number (int)
        [4]  type (int or null)
        [5]  JEL_Code (str or null)
        [6]  Taxonomy (str or null)
        [7]  boardmember (str or null)
        [8]  Authors (str or null)
        [9]  documentTypes — urls list or other
        [10] publicationProperties — metadata dict {Title, Subtitle, ...}
        [11] childrenPublication (list)
        [12] relatedPublications (list)

    Args:
        record: List of 13 field values for one record.

    Returns:
        Dict with title, url, published_at, speaker, or None if invalid.
    """
    if not isinstance(record, list) or len(record) < 11:
        return None

    ts = record[1]

    # URLs can be in field 9 (documentTypes) — look for lists containing URL strings
    urls = []
    for field in record[9:11]:
        if isinstance(field, list):
            for item in field:
                if isinstance(item, str) and ("/" in item):
                    urls.append(item)

    # Metadata with Title is typically in field 10
    metadata = {}
    for field in record[9:13]:
        if isinstance(field, dict):
            metadata = field
            break

    # Get title
    title = metadata.get("Title", "")
    if not title:
        return None

    if not urls:
        return None

    # Find the HTML URL (not PDF)
    url = None
    for u in urls:
        if ".html" in u:
            url = u
            break

    if not url:
        url = urls[0]

    # Build full URL
    if not url.startswith("http"):
        url = ECB_BASE_URL + url

    # Skip PDFs
    if url.endswith(".pdf"):
        return None

    # Parse timestamp
    published_at = datetime.fromtimestamp(ts) if ts else datetime.now()

    # Speaker: try boardmember (field 7), then Authors (field 8)
    speaker = ""
    if record[7]:
        speaker = str(record[7])
    elif record[8]:
        speaker = str(record[8])

    return {
        "title": title,
        "url": url,
        "published_at": published_at,
        "speaker": speaker,
    }


def scrape_speech_content(url: str) -> dict:
    """
    Scrape full text and speaker from an ECB speech page.

    Same logic as ecb_scraper.py but standalone for backfill.
    """
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {"speaker": "ECB", "content": ""}

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract content from main > div.section
    content = ""
    main = soup.find("main")
    if main:
        section = main.find("div", class_="section")
        if section:
            for tag in section.find_all(["script", "style", "nav", "button", "form"]):
                tag.decompose()
            content = section.get_text(separator="\n", strip=True)

    # Extract speaker
    speaker = "ECB"
    if main:
        section = main.find("div", class_="section")
        if section:
            first_p = section.find("p")
            if first_p:
                text = first_p.get_text(strip=True)
                for prefix in [
                    "Speech by ", "Remarks by ", "Keynote by ", "Lecture by ",
                    "Statement by ", "Interview with ", "Introductory statement by ",
                ]:
                    if text.startswith(prefix):
                        name_part = text[len(prefix):]
                        for sep in [",", ".", ";", " at ", " on ", " Member"]:
                            if sep in name_part:
                                speaker = name_part[: name_part.index(sep)].strip()
                                break
                        if speaker == "ECB":
                            speaker = name_part.strip()
                        break

    # Fallback: author meta tag
    if speaker == "ECB":
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            speaker = author_meta["content"]

    return {"speaker": speaker, "content": content}


def fetch_ecb_archive(max_speeches: int = 200, delay: float = 1.0) -> list[ECBArchiveSpeech]:
    """
    Main function: fetch historical ECB speeches from the archive.

    Args:
        max_speeches: Maximum number of speeches to fetch.
        delay: Delay between page scrapes (be respectful to ECB servers).

    Returns:
        List of ECBArchiveSpeech objects.
    """
    logger.info("=" * 60)
    logger.info(f"ECB Archive Backfill — fetching up to {max_speeches} speeches")
    logger.info("=" * 60)

    # Step 1: Get latest database version
    version = get_latest_version()
    if not version:
        logger.error("Could not get database version")
        return []

    timestamp, hash_val = version
    logger.info(f"Database version: {timestamp}/{hash_val}")

    # Step 2: Get all speech row indices
    speech_indices = get_speech_record_ids(timestamp, hash_val)
    if not speech_indices:
        logger.error("No speech indices found")
        return []

    speech_index_set = set(speech_indices)

    # Step 3: Fetch records from data chunks (already parsed into dicts)
    records = get_records_from_chunks(
        timestamp, hash_val, speech_index_set, max_records=max_speeches
    )
    logger.info(f"Found {len(records)} speech records in database")

    # Step 4: Scrape each speech page for full text
    speeches = []
    for i, metadata in enumerate(records):
        logger.info(f"[{i+1}/{len(records)}] Scraping: {metadata['title'][:60]}...")

        page_data = scrape_speech_content(metadata["url"])

        if not page_data["content"]:
            logger.warning(f"  No content found, skipping")
            continue

        # Use page speaker if record didn't have one
        speaker = metadata["speaker"] or page_data["speaker"]

        speech = ECBArchiveSpeech(
            title=metadata["title"],
            speaker=speaker,
            content=page_data["content"],
            published_at=metadata["published_at"],
            url=metadata["url"],
        )
        speeches.append(speech)

        # Be respectful to the ECB server
        time.sleep(delay)

    logger.info(f"Successfully scraped {len(speeches)} ECB archive speeches")
    return speeches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a small batch
    speeches = fetch_ecb_archive(max_speeches=5, delay=1.0)
    for s in speeches:
        print(f"\n{'='*60}")
        print(f"Title:   {s.title[:80]}")
        print(f"Speaker: {s.speaker}")
        print(f"Date:    {s.published_at}")
        print(f"Content: {s.content[:200]}...")
