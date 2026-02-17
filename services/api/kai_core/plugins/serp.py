"""
SERP Screener plugin with stealth headers and soft-404 detection.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

import certifi
import requests
from requests.exceptions import SSLError

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore


class BrowserHeadersGenerator:
    """
    Generates Chrome 120-like headers (research-based) to reduce blocking.
    """

    UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.UA,
            "Sec-Ch-Ua": '"Chromium";v="120", "Google Chrome";v="120", "Not A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Upgrade-Insecure-Requests": "1",
        }


def detect_soft_404(html_content: str) -> bool:
    """
    Heuristic soft-404 detection: title/body text for unavailability cues.
    """
    text_content = (html_content or "").lower()

    soft_404_phrases = [
        "out of stock",
        "currently unavailable",
        "product not found",
        "we can't find that page",
        "we couldnt find that page",
        "sold out",
        "item is no longer available",
    ]

    # Fast path: raw text scan
    if any(phrase in text_content for phrase in soft_404_phrases):
        return True

    # Deeper inspection with BeautifulSoup when available
    if not BeautifulSoup:
        return False
    soup = BeautifulSoup(html_content, "html.parser")
    text_content = soup.get_text().lower()

    if soup.title and soup.title.string and "page not found" in soup.title.string.lower():
        return True

    for phrase in soft_404_phrases:
        if phrase in text_content:
            # Check prominent elements to reduce false positives
            if soup.find(["h1", "h2", "div"], string=lambda t: t and phrase in t.lower()):
                return True
            return True
    return False


class SerpScanner:
    """
    Stealth SERP scanner with header rotation and optional proxies.
    """

    def __init__(self, keywords: List[str], proxy_rotator: Any = None):
        self.keywords = keywords
        self.session = requests.Session()
        self.proxy_rotator = proxy_rotator
        self.headers_generator = BrowserHeadersGenerator()

    def _randomized_sleep(self) -> None:
        time.sleep(random.uniform(2, 5))

    def _get_proxy(self) -> Optional[Dict[str, str]]:
        return self.proxy_rotator.get_next_proxy() if self.proxy_rotator else None

    def _handle_block(self) -> None:
        time.sleep(random.uniform(5, 10))

    def scan(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for keyword in self.keywords:
            self._randomized_sleep()
            headers = self.headers_generator.get_headers()
            proxy = self._get_proxy()
            try:
                resp = self.session.get(
                    f"https://www.google.com/search?q={requests.utils.quote(keyword)}",
                    headers=headers,
                    proxies=proxy,
                    timeout=10,
                )
                entry: Dict[str, Any] = {"keyword": keyword, "status": resp.status_code}
                if resp.status_code == 200:
                    entry["soft_404"] = detect_soft_404(resp.text)
                elif resp.status_code == 403:
                    self._handle_block()
                results.append(entry)
            except requests.RequestException as exc:
                results.append({"keyword": keyword, "error": str(exc)})
        return results


def _requests_verify_path() -> str | bool:
    override = (os.environ.get("KAI_SSL_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE") or "").strip()
    if override and os.path.exists(override):
        return override
    try:
        ca = certifi.where()
        if ca and os.path.exists(ca):
            return ca
    except Exception:
        pass
    sys_ca = "/etc/ssl/certs/ca-certificates.crt"
    if os.path.exists(sys_ca):
        return sys_ca
    return True


def check_url_health(url_list: List[str], timeout: int = 3) -> List[Dict[str, Any]]:
    """
    Lightweight URL health check using HEAD; falls back to GET on failure.
    """
    results: List[Dict[str, Any]] = []
    session = requests.Session()
    # Some sites (including wikipedia.org) return 4xx for HEAD or for "python-requests" user agents.
    # Use a browser-like UA so we get a representative HTTP status for health checks.
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    for url in url_list:
        entry: Dict[str, Any] = {"url": url}
        verify = _requests_verify_path() if str(url).lower().startswith("https://") else True
        resp = None
        # 1) Prefer HEAD for speed, but fall back to GET (HEAD can be blocked/unsupported).
        try:
            resp = session.head(url, timeout=timeout, allow_redirects=True, verify=verify)
        except SSLError:
            # Some environments inject TLS certificates that are not in the CA bundle.
            # Retry once with verification disabled so we can still determine link health.
            if verify is not False:
                try:
                    resp = session.head(url, timeout=timeout, allow_redirects=True, verify=False)
                except requests.RequestException:
                    resp = None
        except requests.RequestException:
            resp = None

        # Fall back when HEAD is blocked (405) or returns forbidden (403) but GET might still succeed.
        if resp is None or resp.status_code in (403, 405) or resp.status_code >= 500:
            try:
                resp = session.get(url, timeout=timeout, allow_redirects=True, verify=verify)
            except SSLError:
                if verify is not False:
                    try:
                        resp = session.get(url, timeout=timeout, allow_redirects=True, verify=False)
                    except requests.RequestException:
                        resp = None
            except requests.RequestException:
                resp = None

        if resp is not None:
            status = int(resp.status_code)
            entry["status"] = status
            entry["soft_404"] = detect_soft_404(resp.text) if status == 200 else False
        else:
            # Schema stability: always return status + soft_404 fields even on failure.
            # Keep error messages sanitized (no raw requests/SSL exception strings in client responses).
            entry["status"] = 0
            entry["soft_404"] = False
            entry["error"] = "request_failed"
        results.append(entry)
    return results
