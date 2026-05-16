#!/usr/bin/env python3
"""
SFU Sitemap Scraper with True Depth Crawling
Scrapes the sitemap and recursively crawls pages up to the specified depth.
"""

import asyncio
import html as html_module
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import urljoin, urlparse

# Project root on path for config
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import requests
from bs4 import BeautifulSoup, Comment

from config import Config

# All output files go to FYP-BackEnd/output
OUTPUT_DIR = os.path.join(_project_root, "output")

# Dedicated thread pool for Crawl4AI (each fetch runs in a thread with its own event loop).
_crawl4ai_executor: Optional[ThreadPoolExecutor] = None
_crawl4ai_executor_lock = threading.Lock()


def _get_crawl4ai_executor() -> ThreadPoolExecutor:
    global _crawl4ai_executor
    with _crawl4ai_executor_lock:
        if _crawl4ai_executor is None:
            _crawl4ai_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="crawl4ai",
            )
        return _crawl4ai_executor


def _run_crawl4ai_sync_on_thread(url: str) -> Optional[str]:
    """Run Crawl4AI in a thread that has no running asyncio loop (safe under asyncio.run(main()))."""
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return None

    async def _arun() -> Optional[str]:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if getattr(result, "success", True) is False:
                return None
            html = getattr(result, "html", None) or getattr(result, "cleaned_html", None) or ""
            if html and len(html) > 100:
                return html
            md = getattr(result, "markdown", None) or ""
            if md:
                esc = html_module.escape(md)
                return (
                    "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>crawl4ai</title></head>"
                    f"<body><pre>{esc}</pre></body></html>"
                )
            return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_arun())
    finally:
        try:
            loop.close()
        except Exception:
            pass
        asyncio.set_event_loop(None)


def _fetch_page_with_crawl4ai(url: str, timeout: float = 180.0) -> Optional[str]:
    """Fetch HTML using Crawl4AI. Always runs in a worker thread to avoid 'loop already running'.

    https://docs.crawl4ai.com/ — requires: pip install crawl4ai && python -m playwright install chromium
    """
    try:
        from crawl4ai import AsyncWebCrawler  # noqa: F401 — verify installed before submit
    except ImportError:
        print("⚠️ crawl4ai not installed. pip install crawl4ai && python -m playwright install chromium")
        return None

    try:
        future = _get_crawl4ai_executor().submit(_run_crawl4ai_sync_on_thread, url)
        return future.result(timeout=timeout)
    except Exception as e:
        print(f"⚠️ Crawl4AI fetch failed for {url[:80]}...: {e}")
        return None


@dataclass
class SitemapItem:
    """Represents a single sitemap entry"""
    title: str
    url: str
    category: str
    subcategory: Optional[str] = None
    level: int = 1


@dataclass
class PageContent:
    """Represents extracted content from a crawled page"""
    url: str
    title: str
    meta_description: Optional[str] = None
    headings: List[Dict[str, str]] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    text_content: str = ""
    word_count: int = 0
    crawl_success: bool = True
    crawl_error: Optional[str] = None
    crawl_timestamp: Optional[str] = None
    crawl_depth: int = 0  # Track at what depth this page was found
    parent_url: Optional[str] = None  # Track which page linked to this


class SFUSitemapScraper:
    """Scraper for SFU sitemap with true recursive depth crawling"""
    
    def __init__(
        self,
        max_workers: int = 10,
        request_delay: float = 0.1,
        max_pages: int = 5000,
        use_crawl4ai: Optional[bool] = None,
    ):
        self.base_url = "https://www.sfu.edu.hk"
        self.sitemap_url = "https://www.sfu.edu.hk/en/site-map/index.html"
        self.items: List[SitemapItem] = []
        self.categories: set = set()
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.max_pages = max_pages  # Safety limit to prevent infinite crawling
        if use_crawl4ai is None:
            self.use_crawl4ai = bool(getattr(Config, "USE_CRAWL4AI", False))
        else:
            self.use_crawl4ai = bool(use_crawl4ai)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        if self.use_crawl4ai:
            print("🌐 Crawl4AI enabled for page fetches (browser). Fallback: requests if a page fails.")
        # Track visited URLs across all depths
        self.visited_urls: Set[str] = set()
        self.all_crawled_pages: List[Dict[str, Any]] = []
        self.url_to_depth: Dict[str, int] = {}  # Track depth for each URL
        
    def normalize_url(self, url: str) -> str:
        """Normalize URL for comparison"""
        # Remove trailing slash, fragments, and normalize
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/').lower()
        return normalized
    
    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to SFU domain"""
        try:
            parsed = urlparse(url)
            return 'sfu.edu.hk' in parsed.netloc.lower()
        except:
            return False

    # File extensions to treat as images (skip crawling)
    _IMAGE_EXTENSIONS = frozenset(
        ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico', '.tiff', '.tif')
    )

    def is_image_url(self, url: str) -> bool:
        """Return True if the URL points to an image file (by path extension)."""
        try:
            parsed = urlparse(url)
            path = (parsed.path or "").lower()
            # Strip query/fragment from path (path doesn't include them, but be safe)
            base = path.split("?")[0]
            return any(base.endswith(ext) for ext in self._IMAGE_EXTENSIONS)
        except Exception:
            return False

    def is_sc_or_tc_url(self, url: str) -> bool:
        """Return True if the URL path contains /sc/ or /tc/ (skip these sites)."""
        try:
            parsed = urlparse(url)
            path = (parsed.path or "").lower()
            return "/sc/" in path or "/tc/" in path
        except Exception:
            return False

    # File extensions to skip (non-HTML; only crawl HTML pages)
    _NON_HTML_EXTENSIONS = frozenset(
        ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.ppt', '.pptx',
         '.csv', '.rtf', '.odt', '.ods', '.odp', '.mp3', '.mp4', '.wav', '.mov')
    )

    def is_non_html_url(self, url: str) -> bool:
        """Return True if the URL points to a non-HTML resource (by path extension)."""
        try:
            parsed = urlparse(url)
            path = (parsed.path or "").lower()
            base = path.split("?")[0]
            return any(base.endswith(ext) for ext in self._NON_HTML_EXTENSIONS)
        except Exception:
            return False

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with error handling. Returns None if response is not HTML."""
        if self.use_crawl4ai:
            html = _fetch_page_with_crawl4ai(url)
            if html:
                return html
            # Fall back to HTTP if browser fetch failed
        try:
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return None
            return response.text
        except requests.exceptions.RequestException as e:
            return None
        except Exception as e:
            return None
    
    def parse_sitemap(self, html: str) -> List[SitemapItem]:
        """Parse the sitemap HTML and extract all links"""
        soup = BeautifulSoup(html, 'html.parser')
        items = []
        seen_urls = set()
        
        # Find the main content area
        main_content = soup.find('main') or soup.find('div', class_='content')
        if not main_content:
            main_content = soup
        
        # Process all links
        for link in main_content.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Skip empty or navigation links
            if not text or not href or href.startswith('#') or href.startswith('javascript'):
                continue
            
            # Skip social media and external links
            if any(x in href for x in ['facebook', 'instagram', 'youtube', 'twitter', 'linkedin']):
                continue
            
            # Make URL absolute
            if href.startswith('/'):
                href = self.base_url + href
            elif not href.startswith('http'):
                href = self.base_url + '/' + href
            
            # Skip duplicates
            if href in seen_urls:
                continue
            seen_urls.add(href)
            
            # Skip non-SFU domains
            if not self.is_same_domain(href):
                continue

            # Skip image files
            if self.is_image_url(href):
                continue
            
            # Skip /sc/ and /tc/ sites
            if self.is_sc_or_tc_url(href):
                continue
            
            # Skip non-HTML resources
            if self.is_non_html_url(href):
                continue
            
            # Determine category based on URL and text
            category, subcategory = self.categorize_link(text, href)
            
            # Determine level
            level = self.determine_level(text, href, category)
            
            item = SitemapItem(
                title=text,
                url=href,
                category=category,
                subcategory=subcategory,
                level=level
            )
            items.append(item)
            self.categories.add(category)
        
        return items
    
    def categorize_link(self, text: str, url: str) -> tuple:
        """Categorize a link based on its URL and text"""
        url_lower = url.lower()
        text_lower = text.lower()
        
        # Define category patterns
        category_patterns = {
            'About': ['/about-the-institute', 'about', 'overview', 'governance', 'milestone'],
            'Admission': ['/admission', 'admission', 'apply', 'application', 'jupas', 'e-app'],
            'Programmes': ['/programmes', 'programme', 'bachelor', 'master', 'diploma', 'degree', 'higher diploma'],
            'Academic Units': ['/schools-and-offices', 'school', 'faculty', 'department', 'centre'],
            'Student Support': ['/student-support', 'student', 'scholarship', 'financial', 'career'],
            'Research': ['/research', 'research', 'funded project'],
            'Donation': ['/donation', 'donate', 'giving'],
            'Global Engagement': ['/gmed', 'global', 'mainland', 'international'],
            'Library': ['/library', 'library'],
            'Career': ['/career', 'career', 'job', 'employment'],
            'News & Media': ['/media', 'news', 'press', 'event'],
            'Alumni': ['/alumni', 'alumni'],
            'Staff': ['/staff', 'staff', 'faculty'],
            'Student': ['/student', 'student life'],
            'Visitor': ['/visitor', 'visitor', 'tour'],
        }
        
        # Check URL patterns first
        for category, patterns in category_patterns.items():
            if any(p in url_lower for p in patterns):
                # Determine subcategory for programmes
                subcategory = None
                if category == 'Programmes':
                    if 'master' in url_lower or 'postgraduate' in text_lower:
                        subcategory = 'Postgraduate'
                    elif 'bachelor' in url_lower or 'undergraduate' in text_lower:
                        subcategory = 'Undergraduate'
                    elif 'higher-diploma' in url_lower or 'sub-degree' in text_lower:
                        subcategory = 'Sub-degree'
                    elif 'professional-diploma' in url_lower or 'diploma' in text_lower:
                        subcategory = 'Professional Diploma'
                
                return category, subcategory
        
        # Default category
        return 'General', None
    
    def determine_level(self, text: str, url: str, category: str) -> int:
        """Determine the hierarchy level of a link"""
        url_parts = url.replace(self.base_url, '').split('/')
        url_parts = [p for p in url_parts if p]
        
        # Level 1: Main category pages (shorter URLs)
        if len(url_parts) <= 2:
            return 1
        
        # Level 2: Subcategory/listing pages
        if 'index' in url or len(url_parts) == 3:
            return 2
        
        # Level 3: Individual pages
        return 3
    
    def extract_page_content(self, url: str, html: str, depth: int = 0, parent_url: Optional[str] = None) -> PageContent:
        """Extract structured content from a page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, and comment elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract meta description
        meta_description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            meta_description = meta_desc.get('content', '')
        
        # Extract headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = h.get_text(strip=True)
            if text and len(text) > 3:
                headings.append({
                    'level': h.name,
                    'text': text
                })
        
        # Extract paragraphs
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Filter out short/empty paragraphs
                paragraphs.append(text)
        
        # Extract links (for next depth level)
        links = []
        seen_links = set()
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            
            # Skip empty, anchor, and javascript links
            if not href or href.startswith('#') or href.startswith('javascript'):
                continue
            
            # Make absolute URL
            if href.startswith('/'):
                href = self.base_url + href
            elif not href.startswith('http'):
                href = urljoin(url, href)
            
            # Skip duplicates
            if href in seen_links:
                continue
            seen_links.add(href)
            
            # Only include SFU domain links
            if not self.is_same_domain(href):
                continue

            # Skip image files
            if self.is_image_url(href):
                continue
            
            # Skip /sc/ and /tc/ sites
            if self.is_sc_or_tc_url(href):
                continue
            
            # Skip non-HTML resources
            if self.is_non_html_url(href):
                continue
            
            links.append({
                'text': text[:100] if text else '',  # Limit text length
                'url': href
            })
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if src:
                if src.startswith('/'):
                    src = self.base_url + src
                elif not src.startswith('http'):
                    src = urljoin(url, src)
                
                images.append({
                    'src': src,
                    'alt': alt
                })
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text(strip=True))
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        
        # Get main content text
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
        if main_content:
            text_content = main_content.get_text(separator='\n', strip=True)
        else:
            text_content = soup.get_text(separator='\n', strip=True)
        
        # Clean up text content
        text_content = re.sub(r'\n+', '\n', text_content)
        text_content = re.sub(r'\s+', ' ', text_content)
        
        # Calculate word count
        word_count = len(text_content.split())
        
        return PageContent(
            url=url,
            title=title,
            meta_description=meta_description,
            headings=headings[:20],  # Limit to top 20 headings
            paragraphs=paragraphs[:30],  # Limit to top 30 paragraphs
            links=links[:100],  # Limit to top 100 links (increased for depth crawling)
            images=images[:20],  # Limit to top 20 images
            tables=tables[:5],  # Limit to top 5 tables
            text_content=text_content[:5000],  # Limit text content
            word_count=word_count,
            crawl_success=True,
            crawl_timestamp=datetime.now().isoformat(),
            crawl_depth=depth,
            parent_url=parent_url
        )
    
    def crawl_single_page(self, url: str, depth: int = 0, parent_url: Optional[str] = None, category: str = "General") -> Dict[str, Any]:
        """Crawl a single page and extract its content"""
        # Normalize URL
        normalized_url = self.normalize_url(url)

        # Skip image files (do not fetch)
        if self.is_image_url(url):
            self.visited_urls.add(normalized_url)
            return {
                'sitemap_item': {'title': 'Skipped (image)', 'url': url, 'category': category, 'subcategory': None, 'level': depth},
                'content': asdict(PageContent(
                    url=url,
                    title='Skipped (image)',
                    crawl_success=False,
                    crawl_error="Skipped: image URL",
                    crawl_timestamp=datetime.now().isoformat(),
                    crawl_depth=depth,
                    parent_url=parent_url
                )),
                'found_links': []
            }
        
        # Skip /sc/ and /tc/ sites
        if self.is_sc_or_tc_url(url):
            self.visited_urls.add(normalized_url)
            return {
                'sitemap_item': {'title': 'Skipped (sc/tc)', 'url': url, 'category': category, 'subcategory': None, 'level': depth},
                'content': asdict(PageContent(
                    url=url,
                    title='Skipped (sc/tc)',
                    crawl_success=False,
                    crawl_error="Skipped: sc/tc path",
                    crawl_timestamp=datetime.now().isoformat(),
                    crawl_depth=depth,
                    parent_url=parent_url
                )),
                'found_links': []
            }
        
        # Skip non-HTML URLs
        if self.is_non_html_url(url):
            self.visited_urls.add(normalized_url)
            return {
                'sitemap_item': {'title': 'Skipped (non-HTML)', 'url': url, 'category': category, 'subcategory': None, 'level': depth},
                'content': asdict(PageContent(
                    url=url,
                    title='Skipped (non-HTML)',
                    crawl_success=False,
                    crawl_error="Skipped: non-HTML URL",
                    crawl_timestamp=datetime.now().isoformat(),
                    crawl_depth=depth,
                    parent_url=parent_url
                )),
                'found_links': []
            }
        
        # Skip if already visited
        if normalized_url in self.visited_urls:
            return None
        
        # Mark as visited
        self.visited_urls.add(normalized_url)
        self.url_to_depth[url] = depth
        
        # Fetch the page
        html = self.fetch_page(url)
        
        if html is None:
            return {
                'sitemap_item': {
                    'title': 'Unknown',
                    'url': url,
                    'category': category,
                    'subcategory': None,
                    'level': depth
                },
                'content': asdict(PageContent(
                    url=url,
                    title='Unknown',
                    crawl_success=False,
                    crawl_error="Failed to fetch page",
                    crawl_timestamp=datetime.now().isoformat(),
                    crawl_depth=depth,
                    parent_url=parent_url
                )),
                'found_links': []
            }
        
        # Extract content
        try:
            page_content = self.extract_page_content(url, html, depth, parent_url)
            # Use URL-derived title if page title is empty
            if not page_content.title:
                page_content.title = url.split('/')[-2].replace('-', ' ').title() if url.endswith('/') else url.split('/')[-1].replace('-', ' ').title()
        except Exception as e:
            return {
                'sitemap_item': {
                    'title': 'Error',
                    'url': url,
                    'category': category,
                    'subcategory': None,
                    'level': depth
                },
                'content': asdict(PageContent(
                    url=url,
                    title='Error',
                    crawl_success=False,
                    crawl_error=str(e),
                    crawl_timestamp=datetime.now().isoformat(),
                    crawl_depth=depth,
                    parent_url=parent_url
                )),
                'found_links': []
            }
        
        # Get links found on this page for next depth level
        found_links = [link['url'] for link in page_content.links]
        
        return {
            'sitemap_item': {
                'title': page_content.title,
                'url': url,
                'category': category,
                'subcategory': None,
                'level': depth
            },
            'content': asdict(page_content),
            'found_links': found_links
        }
    
    def crawl_depth_level(self, urls_to_crawl: List[tuple], current_depth: int) -> List[str]:
        """
        Crawl a single depth level and return URLs for next level
        
        Args:
            urls_to_crawl: List of (url, parent_url, category) tuples
            current_depth: Current crawl depth
        
        Returns:
            List of URLs found for next depth level
        """
        results = []
        next_level_urls = []
        total = len(urls_to_crawl)
        completed = 0
        success_count = 0
        
        print(f"\n🔍 Crawling depth {current_depth}: {total} pages to crawl...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.crawl_single_page, url, current_depth, parent, cat): (url, parent, cat)
                for url, parent, cat in urls_to_crawl
            }
            
            # Process results as they complete
            for future in as_completed(future_to_url):
                url, parent, cat = future_to_url[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        if result['content']['crawl_success']:
                            success_count += 1
                            # Add found links to next level (if not visited)
                            for link in result['found_links']:
                                normalized = self.normalize_url(link)
                                if normalized not in self.visited_urls:
                                    next_level_urls.append((link, url, cat))
                        
                        # Progress update
                        if completed % 10 == 0 or completed == total:
                            print(f"  [{'✅' if result['content']['crawl_success'] else '⚠️'}] [{completed}/{total}] {result['content']['title'][:50]}...")
                        
                except Exception as e:
                    print(f"  ❌ [{completed}/{total}] Error crawling {url}: {e}")
        
        # Add results to overall collection
        self.all_crawled_pages.extend(results)
        
        # Remove duplicates from next level URLs
        seen = set()
        unique_next_urls = []
        for url, parent, cat in next_level_urls:
            normalized = self.normalize_url(url)
            if normalized not in seen and normalized not in self.visited_urls:
                seen.add(normalized)
                unique_next_urls.append((url, parent, cat))
        
        print(f"✅ Depth {current_depth} complete: {success_count}/{total} pages crawled, {len(unique_next_urls)} new URLs found for next level")
        
        return unique_next_urls
    
    def scrape(self, crawl_depth: int = 1) -> Dict[str, Any]:
        """
        Main scraping method with true depth crawling
        
        Args:
            crawl_depth: 0 = sitemap only, 1+ = recursive depth crawling
        """
        print("="*60)
        print("🚀 SFU Sitemap Scraper with True Depth Crawling")
        print("="*60)
        print(f"📍 Sitemap URL: {self.sitemap_url}")
        print(f"🔍 Crawl Depth: {crawl_depth}")
        print(f"⚙️ Max Workers: {self.max_workers}")
        print(f"📄 Max Pages: {self.max_pages}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Step 1: Fetch and parse the sitemap
        print("📋 Phase 1: Parsing sitemap...")
        html = self.fetch_page(self.sitemap_url)
        if not html:
            print("❌ Failed to fetch sitemap")
            return {}
        
        self.items = self.parse_sitemap(html)
        print(f"✅ Found {len(self.items)} unique links in sitemap")
        print(f"📁 Categories: {', '.join(sorted(self.categories))}\n")
        
        # Step 2: Crawl based on depth
        if crawl_depth >= 1:
            # Prepare initial URLs from sitemap (depth 1)
            initial_urls = [(item.url, self.sitemap_url, item.category) for item in self.items]
            
            # Crawl each depth level
            current_urls = initial_urls
            for depth in range(1, crawl_depth + 1):
                if not current_urls:
                    print(f"\n⚠️ No new URLs found at depth {depth}, stopping...")
                    break
                
                if len(self.visited_urls) >= self.max_pages:
                    print(f"\n⚠️ Reached max pages limit ({self.max_pages}), stopping...")
                    break
                
                current_urls = self.crawl_depth_level(current_urls, depth)
                
                # Small delay between depth levels
                if current_urls and depth < crawl_depth:
                    time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        
        # Build output structure
        successful_crawls = [p for p in self.all_crawled_pages if p['content']['crawl_success']]
        failed_crawls = [p for p in self.all_crawled_pages if not p['content']['crawl_success']]
        
        output = {
            "university": "Saint Francis University (SFU)",
            "sitemap_url": self.sitemap_url,
            "scraped_date": datetime.now().isoformat(),
            "crawl_depth": crawl_depth,
            "crawl_stats": {
                "elapsed_time_seconds": round(elapsed_time, 2),
                "total_pages_found": len(self.items) + len([p for p in self.all_crawled_pages if p['content']['crawl_depth'] > 1]),
                "total_pages_crawled": len(self.all_crawled_pages),
                "successful_crawls": len(successful_crawls),
                "failed_crawls": len(failed_crawls),
                "unique_urls": len(self.visited_urls)
            },
            "statistics": {
                "sitemap_items": len(self.items),
                "categories": len(self.categories),
                "pages_crawled": len(successful_crawls),
                "pages_failed": len(failed_crawls),
                "total_words": sum(p['content']['word_count'] for p in successful_crawls)
            },
            "categories": sorted(list(self.categories)),
            "sitemap_items": [asdict(item) for item in self.items],
            "crawled_pages": self.all_crawled_pages
        }
        
        print(f"\n{'='*60}")
        print(f"📊 CRAWLING COMPLETE")
        print(f"{'='*60}")
        print(f"⏱️  Elapsed Time: {elapsed_time:.1f} seconds")
        print(f"📄 Total Pages Crawled: {len(self.all_crawled_pages)}")
        print(f"✅ Successful: {len(successful_crawls)}")
        print(f"❌ Failed: {len(failed_crawls)}")
        print(f"🔗 Unique URLs: {len(self.visited_urls)}")
        print(f"{'='*60}")
        
        return output
    
    def save_data(self, data: Dict[str, Any], filename: str = "sfu_sitemap_crawled.json"):
        """Save scraped data to JSON (output folder: FYP-BackEnd/output)."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Data saved to: {output_path}")
        file_size = os.path.getsize(output_path)
        print(f"📦 File size: {file_size / 1024 / 1024:.2f} MB")
        return output_path


class DataRefiner:
    """Refines scraped data using DeepSeek LLM."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = Config.DEEPSEEK_BASE_URL
        self.model = Config.DEEPSEEK_MODEL
    
    def _parse_json_robust(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM, tolerating trailing commas, single-quoted keys, and truncation."""
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Optional: use json5 for highly lenient parsing (trailing commas, single quotes, etc.)
        try:
            import json5 as json5_lib
            return json5_lib.loads(text)
        except ImportError:
            pass
        except Exception:
            pass
        # Fix single-quoted keys: 'key': -> "key": (identifier-like keys only)
        text = re.sub(r"([,{])\s*'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'\1 "\2":', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Remove trailing commas before } or ]; repeat to handle nested structures
        for _ in range(20):
            prev = text
            text = re.sub(r",(\s*[}\]])", r"\1", text)
            if text == prev:
                break
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
        # Salvage truncated JSON (e.g. "Unterminated string" from max_tokens cut-off)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            err_msg = str(e).lower()
            pos = getattr(e, "pos", None)
            if pos is not None and ("unterminated string" in err_msg or "expecting" in err_msg):
                salvaged = self._salvage_truncated_json(text, pos)
                if salvaged is not None:
                    return salvaged
            raise

    def _salvage_truncated_json(self, text: str, error_pos: int) -> Optional[Dict[str, Any]]:
        """Try to close truncated JSON by cutting at a safe position and adding closers."""
        if error_pos <= 0 or error_pos >= len(text):
            return None
        # Truncate before the error; find last complete key-value (e.g. ", or }, or ],)
        cut = text[:error_pos]
        # Find the last unclosed string (opening " without closing ") and cut before it
        last_safe = cut.rfind('",')
        if last_safe == -1:
            last_safe = cut.rfind('"},')
        if last_safe == -1:
            last_safe = cut.rfind('"],')
        if last_safe != -1:
            cut = cut[: last_safe + 1]
        else:
            cut = text[: min(error_pos, len(cut))]
        # Count unclosed [ and {
        open_braces = cut.count("{") - cut.count("}")
        open_brackets = cut.count("[") - cut.count("]")
        if open_braces < 0 or open_brackets < 0:
            return None
        # Close any open string (we may have cut mid-string)
        if cut.count('"') % 2 != 0:
            cut += '"'
        cut += "]" * open_brackets + "}" * open_braces
        try:
            return json.loads(cut)
        except json.JSONDecodeError:
            return None
    
    async def refine(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine the raw data using DeepSeek."""
        if not self.api_key:
            print("⚠️ No DeepSeek API key provided, skipping AI refinement")
            return self.basic_refinement(raw_data)
        
        try:
            import openai
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            
            print("\n🤖 Refining data with DeepSeek...")
            
            # Prepare a summary of crawled pages for AI
            crawled_pages = raw_data.get("crawled_pages", [])
            page_summaries = []
            
            for page in crawled_pages[:50]:  # Limit to first 50 pages for AI context
                content = page.get('content', {})
                if content.get('crawl_success'):
                    page_summaries.append({
                        'title': content.get('title', ''),
                        'url': content.get('url', ''),
                        'category': page.get('sitemap_item', {}).get('category', ''),
                        'depth': content.get('crawl_depth', 0),
                        'word_count': content.get('word_count', 0),
                        'headings_count': len(content.get('headings', [])),
                        'paragraphs_count': len(content.get('paragraphs', [])),
                        'first_paragraph': content.get('paragraphs', [''])[0][:200] if content.get('paragraphs') else ''
                    })
            
            prompt = f"""
            You are a data refinement expert. I have scraped sitemap data and crawled pages from Saint Francis University (SFU) website.
            
            Please analyze and refine this data with the following improvements:
            
            1. **Content Analysis**: Summarize the main topics/content found on crawled pages
            2. **Page Classification**: Classify each page type (programme_page, info_page, department_page, landing_page, etc.)
            3. **Key Information Extraction**: Extract key information like:
               - Programme details (duration, mode, faculty)
               - Contact information
               - Important dates
               - Requirements
            4. **Content Quality**: Identify pages with rich content vs. sparse content
            5. **Generate Insights**: Provide insights about the website structure and content
            
            Return the refined data in this JSON structure:
            {{
                "university": "Saint Francis University (SFU)",
                "sitemap_url": "https://www.sfu.edu.hk/en/site-map/index.html",
                "scraped_date": "current ISO date",
                "analysis": {{
                    "total_pages_crawled": number,
                    "total_content_words": number,
                    "content_quality_summary": "description of content quality",
                    "main_topics": ["topic1", "topic2", ...],
                    "key_findings": ["finding1", "finding2", ...]
                }},
                "categories_analysis": [
                    {{
                        "name": "category name",
                        "page_count": number,
                        "avg_word_count": number,
                        "content_summary": "brief summary of content in this category",
                        "key_pages": ["page titles"]
                    }}
                ],
                "programmes_summary": [
                    {{
                        "name": "programme name",
                        "level": "undergraduate/postgraduate/etc",
                        "faculty": "faculty name",
                        "url": "url"
                    }}
                ],
                "pages": [
                    {{
                        "title": "page title",
                        "url": "full url",
                        "category": "main category",
                        "page_type": "type of page",
                        "word_count": number,
                        "content_summary": "brief summary of page content",
                        "key_headings": ["heading1", "heading2"]
                    }}
                ]
            }}
            
            Here is the raw data summary:
            - Total sitemap items: {raw_data.get('statistics', {}).get('sitemap_items', 0)}
            - Pages crawled: {raw_data.get('statistics', {}).get('pages_crawled', 0)}
            - Pages failed: {raw_data.get('statistics', {}).get('pages_failed', 0)}
            - Total words: {raw_data.get('statistics', {}).get('total_words', 0)}
            - Categories: {', '.join(raw_data.get('categories', []))}
            
            Page summaries:
            {json.dumps(page_summaries[:30], indent=2)}
            
            Return ONLY valid JSON, no markdown formatting or explanations.
            Keep all string values on a single line (use \\n for newlines). Do not truncate the JSON.
            """
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data refinement expert. Return only valid JSON. Keep every string on one line."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=8192
            )
            
            refined_content = response.choices[0].message.content
            
            # Clean up any markdown formatting
            refined_content = refined_content.strip()
            if refined_content.startswith("```json"):
                refined_content = refined_content[7:]
            if refined_content.startswith("```"):
                refined_content = refined_content[3:]
            if refined_content.endswith("```"):
                refined_content = refined_content[:-3]
            refined_content = refined_content.strip()
            
            refined_data = self._parse_json_robust(refined_content)
            
            # Merge with original data
            refined_data['raw_crawled_data'] = raw_data.get('crawled_pages', [])
            # LLM JSON follows a template with "scraped_date" / no crawl_depth — do not trust those for facts.
            self._merge_scrape_provenance(refined_data, raw_data)

            print("✅ AI refinement complete!")
            return refined_data
            
        except Exception as e:
            print(f"⚠️ AI refinement failed: {e}")
            print("Falling back to basic refinement...")
            return self.basic_refinement(raw_data)

    @staticmethod
    def _merge_scrape_provenance(target: Dict[str, Any], raw_data: Dict[str, Any]) -> None:
        """Copy ground-truth fields from scrape() into refined JSON (LLM output omits or hallucinates them)."""
        if raw_data.get("crawl_depth") is not None:
            target["crawl_depth"] = raw_data["crawl_depth"]
        if raw_data.get("scraped_date"):
            target["scraped_date"] = raw_data["scraped_date"]
        if raw_data.get("crawl_stats") is not None:
            target["crawl_stats"] = raw_data["crawl_stats"]
        if raw_data.get("statistics") is not None:
            target["statistics"] = raw_data["statistics"]
        if raw_data.get("sitemap_url"):
            target["sitemap_url"] = raw_data["sitemap_url"]
        if raw_data.get("university"):
            target["university"] = raw_data["university"]
    
    def basic_refinement(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic refinement without AI"""
        crawled_pages = raw_data.get("crawled_pages", [])
        sitemap_items = raw_data.get("sitemap_items", [])
        
        # Group by category
        categories = {}
        for page in crawled_pages:
            cat = page.get('sitemap_item', {}).get('category', 'General')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(page)
        
        # Calculate category stats
        category_stats = []
        for cat, pages in categories.items():
            successful = [p for p in pages if p['content']['crawl_success']]
            avg_words = sum(p['content']['word_count'] for p in successful) / len(successful) if successful else 0
            category_stats.append({
                'name': cat,
                'page_count': len(pages),
                'successful_crawls': len(successful),
                'avg_word_count': round(avg_words, 0),
                'pages': pages
            })
        
        # Build refined structure
        refined = {
            "university": raw_data.get("university", "Saint Francis University (SFU)"),
            "sitemap_url": raw_data.get("sitemap_url", ""),
            "scraped_date": raw_data.get("scraped_date", datetime.now().isoformat()),
            "crawl_depth": raw_data.get("crawl_depth", 0),
            "statistics": raw_data.get("statistics", {}),
            "crawl_stats": raw_data.get("crawl_stats", {}),
            "categories_analysis": category_stats,
            "all_pages": crawled_pages,
            "sitemap_items": sitemap_items
        }
        
        return refined
    
    def save_refined_data(self, data: Dict[str, Any], filename: str = "sfu_sitemap_refined.json"):
        """Save refined data (output folder: FYP-BackEnd/output)."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Refined data saved to: {output_path}")
        return output_path


def _effective_crawl_depth(data: Dict[str, Any]) -> int:
    """Top-level crawl_depth, or max per-page depth (fixes legacy AI-refined JSON missing the field)."""
    explicit = data.get("crawl_depth")
    pages = data.get("raw_crawled_data") or data.get("crawled_pages") or data.get("all_pages") or []
    depths = [int(p.get("content", {}).get("crawl_depth") or 0) for p in pages if isinstance(p, dict)]
    inferred_max = max(depths) if depths else None
    if explicit is not None:
        ex = int(explicit)
        if ex == 0 and inferred_max is not None and inferred_max > 0:
            return inferred_max
        return ex
    if inferred_max is not None:
        return inferred_max
    return 0


def print_summary(data: Dict[str, Any]):
    """Print a summary of the data"""
    print("\n" + "="*60)
    print("📊 SCRAPING SUMMARY")
    print("="*60)
    
    print(f"🏫 University: {data.get('university', 'N/A')}")
    print(f"📅 Scraped Date: {data.get('scraped_date', 'N/A')}")
    print(f"🔍 Crawl Depth: {_effective_crawl_depth(data)}")
    
    if "crawl_stats" in data:
        stats = data["crawl_stats"]
        print(f"⏱️  Elapsed Time: {stats.get('elapsed_time_seconds', 'N/A')}s")
        print(f"📄 Total Pages Crawled: {stats.get('total_pages_crawled', 'N/A')}")
        print(f"✅ Successful: {stats.get('successful_crawls', 'N/A')}")
        print(f"❌ Failed: {stats.get('failed_crawls', 'N/A')}")
        print(f"🔗 Unique URLs: {stats.get('unique_urls', 'N/A')}")
    
    if "statistics" in data:
        stats = data["statistics"]
        print(f"📝 Total Words: {stats.get('total_words', 'N/A'):,}")
        print(f"📁 Categories: {stats.get('categories', 'N/A')}")
    
    print("="*60 + "\n")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SFU Sitemap Scraper with True Depth Crawling')
    parser.add_argument('--depth', type=int, default=1, help='Crawl depth (0=sitemap only, 1+=recursive)')
    parser.add_argument('--workers', type=int, default=10, help='Number of concurrent workers')
    parser.add_argument('--max-pages', type=int, default=5000, help='Maximum pages to crawl (safety limit)')
    parser.add_argument('--no-ai', action='store_true', help='Skip AI refinement')
    parser.add_argument(
        '--crawl4ai',
        action='store_true',
        help='Use Crawl4AI (headless browser) for page fetches (overrides USE_CRAWL4AI env)',
    )
    args = parser.parse_args()
    
    # Step 1: Scrape the sitemap with depth crawling
    scraper = SFUSitemapScraper(
        max_workers=args.workers,
        max_pages=args.max_pages,
        use_crawl4ai=True if args.crawl4ai else None,
    )
    raw_data = scraper.scrape(crawl_depth=args.depth)
    
    if not raw_data:
        print("❌ Scraping failed")
        return
    
    # Step 2: Save raw data
    raw_file = scraper.save_data(raw_data, f"sfu_sitemap_depth{args.depth}_raw.json")
    
    # Step 3: Refine with AI (unless disabled)
    if not args.no_ai:
        refiner = DataRefiner()
        refined_data = await refiner.refine(raw_data)
        
        # Step 4: Save refined data
        refined_file = refiner.save_refined_data(refined_data, f"sfu_sitemap_depth{args.depth}_refined.json")
    else:
        refined_data = raw_data
    
    # Step 5: Print summary
    print_summary(refined_data)
    
    print("✨ Scraping complete!")
    print("📁 Output files:")
    print(f"   • sfu_sitemap_depth{args.depth}_raw.json - Raw crawled data")
    if not args.no_ai:
        print(f"   • sfu_sitemap_depth{args.depth}_refined.json - AI-refined data")


if __name__ == "__main__":
    asyncio.run(main())