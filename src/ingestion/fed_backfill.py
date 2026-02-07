"""
Federal Reserve Historical Backfill Scraper
---------------------------------------------
Fetches historical Fed speeches from yearly archive pages.

The Fed organizes speeches by year at:
    https://www.federalreserve.gov/newsevents/speech/{year}-speeches.htm

Each page lists all speeches for that year with:
    - Date (<time> element)
    - Title (link to full speech)
    - Speaker name (paragraph text)
    - Location (paragraph text)

Structure per entry:
    <div class="row">
        <div class="col-xs-3 col-md-2 eventlist__time">
            <time>MM/DD/YYYY</time>
        </div>
        <div class="col-xs-9 col-md-10 eventlist__event">
            <p><a href="/newsevents/speech/..."><em>Title</em></a></p>
            <p>Speaker Name</p>
            <p>Location</p>
        </div>
    </div>
"""

import logging
import time
import urllib3
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

FED_BASE_URL = "https://www.federalreserve.gov"
FED_ARCHIVE_URL = "https://www.federalreserve.gov/newsevents/speech/{year}-speeches.htm"


@dataclass
class FedArchiveSpeech:
    """Represents a speech from the Fed archive."""

    title: str
    speaker: str
    content: str
    published_at: datetime
    url: str
    location: str = ""
    source: str = "fed"


def fetch_year_index(year: int) -> list[dict]:
    """
    Fetch the list of speeches for a given year.

    Args:
        year: The year to fetch (e.g., 2025).

    Returns:
        List of dicts with title, speaker, url, date, location.
    """
    url = FED_ARCHIVE_URL.format(year=year)
    logger.info(f"Fetching Fed speech index for {year}: {url}")

    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    entries = []
    rows = soup.find_all("div", class_="row")

    for row in rows:
        # Find date
        time_el = row.find("time")
        if not time_el:
            continue

        date_str = time_el.get_text(strip=True)

        # Find event details
        event_div = row.find("div", class_="eventlist__event")
        if not event_div:
            continue

        # Find title link
        title_link = event_div.find("a", href=lambda h: h and "/newsevents/speech/" in h)
        if not title_link:
            continue

        title = title_link.get_text(strip=True)
        href = title_link.get("href", "")

        if not href.startswith("http"):
            href = FED_BASE_URL + href

        # Find speaker and location from paragraphs
        paragraphs = event_div.find_all("p")
        speaker = ""
        location = ""

        for p in paragraphs:
            text = p.get_text(strip=True)
            # Skip if it's the title link or watch live link
            if p.find("a"):
                continue
            # Speaker lines typically contain titles like Governor, Chair, Vice Chair
            if any(kw in text for kw in ["Governor", "Chair", "Vice Chair", "President"]):
                speaker = text
            elif text and not speaker:
                # Could be speaker without formal title
                speaker = text
            elif text and speaker:
                location = text

        # Parse date
        try:
            published_at = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            logger.warning(f"Could not parse date: {date_str}")
            continue

        entries.append({
            "title": title,
            "speaker": speaker,
            "url": href,
            "published_at": published_at,
            "location": location,
        })

    logger.info(f"Found {len(entries)} speeches for {year}")
    return entries


def scrape_speech_content(url: str) -> str:
    """
    Scrape the full text of a speech from the Fed website.

    Same logic as fed_scraper.py but standalone for backfill.
    """
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Try: article div (most common for speeches)
    article = soup.find("div", id="article")
    if article:
        for tag in article.find_all(["script", "style", "nav"]):
            tag.decompose()
        return article.get_text(separator="\n", strip=True)

    # Try: main content column
    col = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
    if col:
        for tag in col.find_all(["script", "style", "nav"]):
            tag.decompose()
        return col.get_text(separator="\n", strip=True)

    # Try: generic article tag
    article_tag = soup.find("article")
    if article_tag:
        for tag in article_tag.find_all(["script", "style", "nav"]):
            tag.decompose()
        return article_tag.get_text(separator="\n", strip=True)

    # Fallback: main content area
    main = soup.find("main")
    if main:
        for tag in main.find_all(["script", "style", "nav"]):
            tag.decompose()
        return main.get_text(separator="\n", strip=True)

    return ""


def fetch_fed_archive(
    years: list[int] | None = None,
    max_speeches_per_year: int | None = None,
    delay: float = 1.0,
) -> list[FedArchiveSpeech]:
    """
    Main function: fetch historical Fed speeches from archive pages.

    Args:
        years: List of years to scrape. Defaults to [2024, 2025].
        max_speeches_per_year: Limit per year. None = all.
        delay: Delay between page scrapes (be respectful).

    Returns:
        List of FedArchiveSpeech objects.
    """
    if years is None:
        years = [2024, 2025]

    logger.info("=" * 60)
    logger.info(f"Fed Archive Backfill — years: {years}")
    logger.info("=" * 60)

    all_speeches = []

    for year in years:
        entries = fetch_year_index(year)

        if max_speeches_per_year:
            entries = entries[:max_speeches_per_year]

        for i, entry in enumerate(entries):
            logger.info(
                f"[{year}] [{i+1}/{len(entries)}] Scraping: {entry['title'][:60]}..."
            )

            content = scrape_speech_content(entry["url"])

            if not content:
                logger.warning(f"  No content found, skipping")
                continue

            speech = FedArchiveSpeech(
                title=entry["title"],
                speaker=entry["speaker"],
                content=content,
                published_at=entry["published_at"],
                url=entry["url"],
                location=entry.get("location", ""),
            )
            all_speeches.append(speech)

            # Be respectful
            time.sleep(delay)

    logger.info(f"Successfully scraped {len(all_speeches)} Fed archive speeches")
    return all_speeches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with small batch
    speeches = fetch_fed_archive(years=[2025], max_speeches_per_year=5, delay=1.0)
    for s in speeches:
        print(f"\n{'='*60}")
        print(f"Title:   {s.title[:80]}")
        print(f"Speaker: {s.speaker}")
        print(f"Date:    {s.published_at}")
        print(f"Content: {s.content[:200]}...")
