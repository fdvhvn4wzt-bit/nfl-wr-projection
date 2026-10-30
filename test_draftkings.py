"""Standalone, read-only DraftKings public HTML availability diagnostic.

Does not log in, bypass access restrictions, or change your NFL app.
Run: python test_draftkings.py
"""
import html
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

URL = 'https://sportsbook.draftkings.com/page/nfl-player-props?category=receiving-props&subcategory=receiving-yards'

class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.skip += 1
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript') and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())

def main():
    print('DraftKings public page HTML test')
    print('Time (UTC):', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    print('URL:', URL)
    request = urllib.request.Request(URL, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; NFLPropsDiagnostic/1.0)',
        'Accept': 'text/html,application/xhtml+xml',
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
            content_type = response.headers.get('Content-Type', '')
            raw = response.read(2_000_000)
    except urllib.error.HTTPError as error:
        print('HTTP status:', error.code)
        print('The site did not provide a usable page to this request.')
        print('Do not try to bypass site restrictions; keep your Odds API integration.')
        return
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print('Connection failed:', type(error).__name__)
        print('Details:', str(error)[:250])
        return

    document = raw.decode('utf-8', errors='replace')
    parser = VisibleText()
    parser.feed(document)
    visible = html.unescape(' '.join(parser.parts))
    print('HTTP status:', status)
    print('Content type:', content_type)
    print('Downloaded bytes:', len(raw))
    print('Visible text characters:', len(visible))
    print('Page title:', (re.search(r'<title[^>]*>(.*?)</title>', document, re.I | re.S) or [None, 'Not found'])[1].strip()[:120])
    print('Visible text preview:', visible[:500] or '(none)')
    print('Example player names present in raw HTML:')
    for name in ('Zay Flowers', 'Dalton Kincaid', 'Khalil Shakir', 'Mark Andrews'):
        print('  {}: {}'.format(name, 'YES' if name.casefold() in html.unescape(document).casefold() else 'NO'))
    print('Receiving yards phrase in HTML:', 'YES' if 'receiving yards' in document.casefold() else 'NO')
    print('\nInterpretation:')
    print('A 200 response alone does not mean the actual odds are available to Python.')
    print('If player names are absent, the odds are probably loaded separately in the browser.')
    print('Even if names are present, this test does NOT verify standard over/under lines,')
    print('data accuracy, reliability, or permission to automate sportsbook collection.')

if __name__ == '__main__':
    main()
