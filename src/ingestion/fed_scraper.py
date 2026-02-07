"""
Federal Reserve Speech Scraper
-------------------------------
Fetches speeches and statements from the Federal Reserve using their
JSON endpoint and RSS feeds, then scrapes full text from linked pages.

Sources:
    - JSON: https://www.federalreserve.gov/json/ne-speeches.json
    - RSS:  https://www.federalreserve.gov/feeds/speeches.xml
"""

import logging
import ssl
import urllib3
from dataclasses import dataclass
from datetime import datetime
from urllib.request import Request, urlopen

import feedparser
import requests
from bs4 import BeautifulSoup

# Suppress SSL warnings — needed for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

FED_JSON_URL = "https://www.federalreserve.gov/json/ne-speeches.json"
FED_RSS_URL = "https://www.federalreserve.gov/feeds/speeches.xml"
FED_BASE_URL = "https://www.federalreserve.gov"


@dataclass
class FedSpeech:
    """Represents a single Federal Reserve speech."""

    title: str
    speaker: str
    content: str
    published_at: datetime
    url: str
    location: str = ""
    source: str = "fed"


def fetch_speeches_json(url: str = FED_JSON_URL) -> list[dict]:
    """
    Fetch speeches from the Fed's JSON endpoint.

    The JSON endpoint provides structured data with separate speaker field.

    Returns:
        List of speech metadata dicts.
    """
    logger.info(f"Fetching Fed speeches from JSON endpoint: {url}")

    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Fed JSON: {e}")
        return []
    except ValueError as e:
        logger.error(f"Failed to parse Fed JSON: {e}")
        return []

    entries = []
    for item in data:
        link = item.get("l", "")
        if link and not link.startswith("http"):
            link = FED_BASE_URL + link

        entries.append(
            {
                "title": item.get("t", ""),
                "speaker": item.get("s", ""),
                "link": link,
                "location": item.get("lo", ""),
                "date": item.get("d", ""),
            }
        )

    logger.info(f"Found {len(entries)} entries in Fed JSON feed")
    return entries


def fetch_speeches_rss(url: str = FED_RSS_URL) -> list[dict]:
    """
    Fallback: Fetch speeches from the Fed RSS feed.

    Used if the JSON endpoint is unavailable.

    Returns:
        List of speech metadata dicts.
    """
    logger.info(f"Fetching Fed speeches from RSS: {url}")

    # Fetch manually to handle SSL issues
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urlopen(req, context=ssl_context, timeout=30)
        rss_content = response.read()
        feed = feedparser.parse(rss_content)
    except Exception as e:
        logger.error(f"Failed to fetch RSS feed: {e}")
        feed = feedparser.parse(url)

    entries = []
    for entry in feed.entries:
        title = entry.get("title", "")

        # RSS title format: "LastName, Speech Title"
        speaker = ""
        if ", " in title:
            speaker = title.split(", ")[0].strip()
            title = ", ".join(title.split(", ")[1:]).strip()

        entries.append(
            {
                "title": title,
                "speaker": speaker,
                "link": entry.get("link", ""),
                "location": entry.get("summary", ""),
                "date": entry.get("published", ""),
                "published_parsed": entry.get("published_parsed"),
            }
        )

    logger.info(f"Found {len(entries)} entries in Fed RSS feed")
    return entries


def parse_fed_date(date_str: str) -> datetime:
    """
    Parse date from Fed JSON or RSS format.

    Args:
        date_str: Date string (e.g., '01/15/2025' or 'Mon, 15 Jan 2025 12:00:00 GMT')

    Returns:
        Parsed datetime object.
    """
    # JSON format: MM/DD/YYYY
    for fmt in ["%m/%d/%Y", "%B %d, %Y", "%a, %d %b %Y %H:%M:%S %Z"]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    logger.warning(f"Could not parse date: {date_str}")
    return datetime.now()


def scrape_speech_page(url: str) -> str:
    """
    Scrape the full text of a speech from the Fed website.

    Args:
        url: Full URL of the speech page.

    Returns:
        Full text content of the speech.
    """
    logger.info(f"Scraping Fed page: {url}")

    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Fed speeches: main content in div#article or div.col-xs-12
    content = _extract_content(soup)

    return content


def _extract_content(soup: BeautifulSoup) -> str:
    """
    Extract the main text content from a Fed speech page.

    Tries multiple selectors to handle different page layouts.
    """
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

    logger.warning("Could not find main content area")
    return ""


def fetch_fed_speeches() -> list[FedSpeech]:
    """
    Main function: fetch all recent Fed speeches.

    Tries the JSON endpoint first, falls back to RSS.

    Returns:
        List of FedSpeech objects with full text and metadata.
    """
    # Try JSON first (better structured data)
    entries = fetch_speeches_json()

    # Fallback to RSS
    if not entries:
        logger.info("JSON endpoint failed, falling back to RSS")
        entries = fetch_speeches_rss()

    speeches = []

    for entry in entries:
        url = entry["link"]

        if not url:
            logger.warning(f"No URL for entry: {entry['title']}")
            continue

        # Scrape full page
        content = scrape_speech_page(url)

        # Skip entries with no content
        if not content:
            logger.warning(f"Skipping entry with no content: {entry['title']}")
            continue

        speech = FedSpeech(
            title=entry["title"],
            speaker=entry["speaker"],
            content=content,
            published_at=parse_fed_date(entry["date"]),
            url=url,
            location=entry.get("location", ""),
        )
        speeches.append(speech)
        logger.info(f"Scraped: {speech.title} by {speech.speaker}")

    logger.info(f"Successfully scraped {len(speeches)} Fed speeches")
    return speeches


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    speeches = fetch_fed_speeches()
    for s in speeches[:3]:
        print(f"\n{'='*60}")
        print(f"Title:     {s.title}")
        print(f"Speaker:   {s.speaker}")
        print(f"Date:      {s.published_at}")
        print(f"Location:  {s.location}")
        print(f"URL:       {s.url}")
        print(f"Content:   {s.content[:200]}...")
