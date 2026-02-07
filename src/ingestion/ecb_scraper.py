"""
ECB Speech & Press Release Scraper
-----------------------------------
Fetches speeches, press releases, and interviews from the ECB RSS feed,
then scrapes the full text from each linked page.

Source: https://www.ecb.europa.eu/rss/press.html
"""

import logging
import ssl
from dataclasses import dataclass
from datetime import datetime
from urllib.request import Request, urlopen

import feedparser
import requests
import urllib3
from bs4 import BeautifulSoup

# Suppress SSL warnings — needed for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

ECB_RSS_URL = "https://www.ecb.europa.eu/rss/press.html"
ECB_BASE_URL = "https://www.ecb.europa.eu"


@dataclass
class ECBSpeech:
    """Represents a single ECB speech/press release."""

    title: str
    speaker: str
    content: str
    published_at: datetime
    url: str
    source: str = "ecb"


def fetch_rss_feed(url: str = ECB_RSS_URL) -> list[dict]:
    """
    Fetch and parse the ECB RSS feed.

    Returns:
        List of feed entries with title, link, and published date.
    """
    logger.info(f"Fetching ECB RSS feed from {url}")

    # Fetch the RSS content manually to handle SSL issues (corporate proxies, etc.)
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
        feed = feedparser.parse(url)  # Fallback to default feedparser behavior

    if feed.bozo:
        logger.warning(f"RSS feed parsing issue: {feed.bozo_exception}")

    entries = []
    for entry in feed.entries:
        entries.append(
            {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "published_parsed": entry.get("published_parsed"),
            }
        )

    logger.info(f"Found {len(entries)} entries in ECB RSS feed")
    return entries


def parse_published_date(entry: dict) -> datetime:
    """
    Parse the published date from an RSS entry.

    Args:
        entry: RSS entry dict with 'published_parsed' or 'published' fields.

    Returns:
        Parsed datetime object.
    """
    if entry.get("published_parsed"):
        from time import mktime

        return datetime.fromtimestamp(mktime(entry["published_parsed"]))

    # Fallback: try parsing the raw string
    try:
        return datetime.strptime(entry["published"], "%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, KeyError):
        logger.warning(f"Could not parse date for entry: {entry.get('title')}")
        return datetime.now()


def scrape_speech_page(url: str) -> dict:
    """
    Scrape the full text and speaker from an ECB speech/press release page.

    Args:
        url: Full URL of the ECB page.

    Returns:
        Dict with 'speaker' and 'content' keys.
    """
    logger.info(f"Scraping ECB page: {url}")

    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {"speaker": "Unknown", "content": ""}

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract speaker — ECB pages typically have speaker in a subtitle or author element
    speaker = _extract_speaker(soup)

    # Extract main content
    content = _extract_content(soup)

    return {"speaker": speaker, "content": content}


def _extract_speaker(soup: BeautifulSoup) -> str:
    """
    Extract the speaker name from an ECB page.

    Current ECB structure:
        <main>
            <div class="title"> ... </div>
            <div class="section"> Speech by [Name], ... </div>
        </main>

    The speaker name is in the first line of div.section inside <main>.
    """
    main = soup.find("main")
    if main:
        section = main.find("div", class_="section")
        if section:
            # First paragraph often starts with "Speech by Name, Member of..."
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
                                return name_part[: name_part.index(sep)].strip()
                        return name_part.strip()

    # Try: author meta tag
    author_meta = soup.find("meta", attrs={"name": "author"})
    if author_meta and author_meta.get("content"):
        return author_meta["content"]

    return "ECB"


def _extract_content(soup: BeautifulSoup) -> str:
    """
    Extract the main text content from an ECB page.

    Current ECB structure:
        <main>
            <div class="title"> ... </div>
            <div class="section"> [actual speech content] </div>
            <div class="footnotes"> ... </div>
            <div class="related-topics"> ... </div>
        </main>

    The speech text is in <div class="section"> inside <main>.
    The search filters ("ANYTIME", "PAST MONTH") are NOT inside <main>.
    """
    main = soup.find("main")
    if main:
        # Primary: div.section inside main (speech body)
        section = main.find("div", class_="section")
        if section:
            for tag in section.find_all(["script", "style", "nav", "button", "form"]):
                tag.decompose()
            return section.get_text(separator="\n", strip=True)

        # Fallback: all content divs inside main (excluding navigation noise)
        for tag in main.find_all(["script", "style", "nav", "button", "form"]):
            tag.decompose()
        # Remove non-content divs
        for tag in main.find_all("div", class_=["related-topics", "address-box"]):
            tag.decompose()
        return main.get_text(separator="\n", strip=True)

    logger.warning("Could not find <main> element")
    return ""


def fetch_ecb_speeches() -> list[ECBSpeech]:
    """
    Main function: fetch all recent ECB speeches from the RSS feed.

    Returns:
        List of ECBSpeech objects with full text and metadata.
    """
    entries = fetch_rss_feed()
    speeches = []

    for entry in entries:
        url = entry["link"]

        # Ensure full URL
        if url.startswith("/"):
            url = ECB_BASE_URL + url

        # Scrape full page
        page_data = scrape_speech_page(url)

        # Skip entries with no content
        if not page_data["content"]:
            logger.warning(f"Skipping entry with no content: {entry['title']}")
            continue

        speech = ECBSpeech(
            title=entry["title"],
            speaker=page_data["speaker"],
            content=page_data["content"],
            published_at=parse_published_date(entry),
            url=url,
        )
        speeches.append(speech)
        logger.info(f"Scraped: {speech.title} by {speech.speaker}")

    logger.info(f"Successfully scraped {len(speeches)} ECB speeches")
    return speeches


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    speeches = fetch_ecb_speeches()
    for s in speeches[:3]:
        print(f"\n{'='*60}")
        print(f"Title:     {s.title}")
        print(f"Speaker:   {s.speaker}")
        print(f"Date:      {s.published_at}")
        print(f"URL:       {s.url}")
        print(f"Content:   {s.content[:200]}...")
