#!/usr/bin/env python3
"""
EPUB Fixer & Translator - Fix and translate Chinese EPUBs in one step

Uses lxml for proper XHTML parsing and serialization (like Calibre).
Translates Chinese to English using Google Translate (Free).

Usage: 
    python fixTranslate.py novel.epub                  # Full pipeline (fix + translate)
    python fixTranslate.py novel.epub --no-translate   # Fix only
"""

import sys
import zipfile
import tempfile
import os
import re
import argparse
import time
import subprocess
import shutil
import unicodedata
import concurrent.futures
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from io import BytesIO

# Check for required dependencies
try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

try:
    from lxml import etree
    from lxml.html import HtmlElement
except ImportError:
    print("Error: 'lxml' library required. Install with: pip install lxml")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================
XHTML_NS = 'http://www.w3.org/1999/xhtml'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
XHTML = lambda name: f'{{{XHTML_NS}}}{name}'

# Tags that should NOT be self-closing in EPUB output (from Calibre)
SELF_CLOSING_BAD_TAGS = {
    'a', 'abbr', 'address', 'article', 'aside', 'audio', 'b',
    'bdo', 'blockquote', 'body', 'button', 'cite', 'code', 'dd', 'del', 'details',
    'dfn', 'div', 'dl', 'dt', 'em', 'fieldset', 'figcaption', 'figure', 'footer',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hgroup', 'i', 'iframe', 'ins', 'kbd',
    'label', 'legend', 'li', 'map', 'mark', 'meter', 'nav', 'ol', 'output', 'p',
    'pre', 'progress', 'q', 'rp', 'rt', 'samp', 'section', 'select', 'small',
    'span', 'strong', 'sub', 'summary', 'sup', 'textarea', 'time', 'ul', 'var',
    'video', 'title', 'script', 'style'
}

# Elements to remove entirely
REMOVE_ELEMENTS = {'script', 'embed', 'object', 'form', 'input', 'button', 'textarea'}

# Characters to clean for e-reader compatibility
INVISIBLE_CHARS = '\u200b\u200c\u200d\ufeff\u00ad\u2060\u180e\u200e\u200f\u202a\u202b\u202c\u202d\u202e'

DEFAULT_WATERMARKS = [
    r'本書由.{0,30}首發', r'本文由.{0,30}首發', r'正版請.{0,30}閱讀',
    r'請到.{0,30}閱讀', r'最新章節.{0,30}閱讀', r'手機閱讀.{0,50}',
    r'訪問下載.{0,50}', r'更多精彩.{0,50}', r'歡迎廣大書友.{0,50}',
    r'喜歡請收藏.{0,50}', r'請記住本書.{0,50}', r'百度搜索.{0,50}',
    r'最快更新.{0,50}', r'無彈窗.{0,30}',
    r'[𝕒-𝕫𝔸-𝕫𝟘-𝟡]+\.[𝕒-𝕫𝔸-𝕫]+',
    r'[ａ-ｚＡ-Ｚ０-９]+\.[ａ-ｚＡ-Ｚ]+',
    r'關注公眾號.{0,50}', r'微信公眾號.{0,50}', r'掃碼關注.{0,50}',
    r'點擊下載.{0,50}', r'APP下載.{0,50}',
]


# ============================================================================
# GOOGLE TRANSLATE (FREE) - Concurrent implementation
# ============================================================================
class GoogleFreeTranslate:
    """Google Translate Free API with concurrent requests."""
    
    ENDPOINT = 'https://translate.googleapis.com/translate_a/single'
    USER_AGENT = (
        'DeepLBrowserExtension/1.3.0 Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    )
    
    def __init__(self, source_lang='zh-CN', target_lang='en', max_workers=0,
                 request_timeout=10, request_attempt=3, request_interval=0.0, verbose=True):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.request_attempt = request_attempt
        self.request_interval = request_interval
        self.verbose = verbose
        
        self.stats = {'requests': 0, 'paragraphs_translated': 0, 
                      'characters_translated': 0, 'cache_hits': 0, 'errors': 0}
        self.cache: Dict[str, str] = {}
        self.cache_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.completed = 0
        self.total = 0
    
    def _translate_single(self, text: str, index: int) -> Tuple[int, str]:
        if not text or not text.strip():
            return (index, text)
        
        cache_key = text.strip()
        with self.cache_lock:
            if cache_key in self.cache:
                with self.stats_lock:
                    self.stats['cache_hits'] += 1
                self._update_progress()
                return (index, self.cache[cache_key])
        
        params = {'client': 'gtx', 'sl': self.source_lang, 'tl': self.target_lang,
                  'dt': 't', 'dj': '1', 'q': text}
        
        for attempt in range(self.request_attempt):
            try:
                if len(text) <= 1800:
                    response = requests.get(self.ENDPOINT, params=params,
                        headers={'User-Agent': self.USER_AGENT}, timeout=self.request_timeout)
                else:
                    response = requests.post(self.ENDPOINT, data=params,
                        headers={'User-Agent': self.USER_AGENT}, timeout=self.request_timeout)
                
                response.raise_for_status()
                data = response.json()
                translated = ''.join(s.get('trans', '') for s in data.get('sentences', []) if 'trans' in s)
                
                with self.cache_lock:
                    self.cache[cache_key] = translated
                with self.stats_lock:
                    self.stats['requests'] += 1
                    self.stats['paragraphs_translated'] += 1
                    self.stats['characters_translated'] += len(text)
                
                self._update_progress()
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
                return (index, translated)
                
            except Exception:
                if attempt < self.request_attempt - 1:
                    time.sleep(5 * (attempt + 1))
        
        with self.stats_lock:
            self.stats['errors'] += 1
        self._update_progress()
        return (index, text)
    
    def _update_progress(self):
        with self.progress_lock:
            self.completed += 1
            if self.verbose and self.total > 0:
                pct = (self.completed / self.total) * 100
                print(f"\r  Translating: {self.completed}/{self.total} ({pct:.1f}%)", end='', flush=True)
    
    def translate_concurrent(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        
        self.total = len(texts)
        self.completed = 0
        workers = self.max_workers if self.max_workers > 0 else min(len(texts), 100)
        
        if self.verbose:
            print(f"  Starting translation with {workers} concurrent workers...")
        
        results = [''] * len(texts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._translate_single, text, i): i for i, text in enumerate(texts)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, translated = future.result()
                    results[index] = translated
                except Exception:
                    index = futures[future]
                    results[index] = texts[index]
        
        if self.verbose:
            print()
        return results


# ============================================================================
# XHTML PROCESSOR - Using lxml like Calibre
# ============================================================================
class XHTMLProcessor:
    """
    Process XHTML files using lxml for proper parsing and serialization.
    Based on Calibre's approach for maximum e-reader compatibility.
    """
    
    def __init__(self, watermark_patterns=None, verbose=True):
        self.verbose = verbose
        self.watermark_patterns = []
        for pattern in (watermark_patterns or DEFAULT_WATERMARKS):
            try:
                self.watermark_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass
        
        self.stats = {
            'elements_removed': 0, 'empty_tags_removed': 0,
            'watermarks_removed': 0, 'chars_cleaned': 0,
            'self_closing_fixed': 0
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text content - remove watermarks and invisible chars."""
        if not text:
            return text
        
        original = text
        
        # Remove invisible characters
        for char in INVISIBLE_CHARS:
            text = text.replace(char, '')
        
        # Replace non-breaking hyphen
        text = text.replace('\u2011', '-')
        
        # Remove watermarks
        for pattern in self.watermark_patterns:
            text = pattern.sub('', text)
        
        if text != original:
            self.stats['chars_cleaned'] += 1
        
        return text
    
    def parse_xhtml(self, data: bytes, filename: str = '<string>') -> Optional[etree._Element]:
        """Parse XHTML/HTML data into an lxml element tree."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Try to decode
        for encoding in ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1']:
            try:
                text = data.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            return None
        
        # Remove null bytes
        text = text.replace('\0', '')
        
        # Strip encoding declarations that might conflict
        text = re.sub(r'<\?xml[^>]*\?>', '', text)
        text = re.sub(r'encoding\s*=\s*["\'][^"\']*["\']', '', text)
        
        try:
            # Try parsing as XML first
            parser = etree.XMLParser(recover=True, no_network=True)
            root = etree.fromstring(text.encode('utf-8'), parser)
        except Exception:
            try:
                # Fall back to HTML parser
                from lxml import html
                root = html.fromstring(text)
                # Convert to proper XHTML
                if root.tag == 'html' or root.tag == XHTML('html'):
                    pass
                else:
                    # Wrap in html/body if needed
                    new_root = etree.Element(XHTML('html'))
                    body = etree.SubElement(new_root, XHTML('body'))
                    body.append(root)
                    root = new_root
            except Exception:
                return None
        
        return root
    
    def fix_structure(self, root: etree._Element) -> etree._Element:
        """Fix XHTML structure for e-reader compatibility."""
        
        # Ensure we're in XHTML namespace
        if root.tag == 'html':
            root.tag = XHTML('html')
        
        # Ensure all children are in XHTML namespace
        for elem in root.iter():
            if isinstance(elem.tag, str) and not elem.tag.startswith('{'):
                elem.tag = XHTML(elem.tag)
        
        # Ensure <head> exists
        head = root.find(f'.//{{{XHTML_NS}}}head')
        if head is None:
            head = root.find('.//head')
        if head is None:
            head = etree.Element(XHTML('head'))
            root.insert(0, head)
        elif head.tag == 'head':
            head.tag = XHTML('head')
        
        # Ensure <title> exists in head
        title = head.find(f'{{{XHTML_NS}}}title')
        if title is None:
            title = head.find('title')
        if title is None:
            title = etree.SubElement(head, XHTML('title'))
            title.text = 'Unknown'
        elif title.tag == 'title':
            title.tag = XHTML('title')
        if not title.text or not title.text.strip():
            title.text = 'Unknown'
        
        # Ensure proper meta charset
        for meta in head.findall(f'.//{{{XHTML_NS}}}meta[@http-equiv]'):
            if meta.get('http-equiv', '').lower() == 'content-type':
                meta.getparent().remove(meta)
        for meta in head.findall('.//meta[@http-equiv]'):
            if meta.get('http-equiv', '').lower() == 'content-type':
                meta.getparent().remove(meta)
        
        meta = etree.Element(XHTML('meta'))
        meta.set('http-equiv', 'Content-Type')
        meta.set('content', 'text/html; charset=utf-8')
        head.insert(0, meta)
        
        # Ensure <body> exists
        body = root.find(f'.//{{{XHTML_NS}}}body')
        if body is None:
            body = root.find('.//body')
        if body is None:
            body = etree.SubElement(root, XHTML('body'))
        elif body.tag == 'body':
            body.tag = XHTML('body')
        
        return root
    
    def clean_content(self, root: etree._Element) -> etree._Element:
        """Clean content - remove bad elements, fix text, etc."""
        
        # Remove forbidden elements
        for tag in REMOVE_ELEMENTS:
            for elem in root.findall(f'.//{{{XHTML_NS}}}{tag}'):
                self._remove_element_keep_tail(elem)
                self.stats['elements_removed'] += 1
            for elem in root.findall(f'.//{tag}'):
                self._remove_element_keep_tail(elem)
                self.stats['elements_removed'] += 1
        
        # Convert deprecated tags
        conversions = [
            ('center', 'div', {'style': 'text-align:center'}),
            ('u', 'span', {'style': 'text-decoration:underline'}),
            ('s', 'span', {'style': 'text-decoration:line-through'}),
            ('strike', 'span', {'style': 'text-decoration:line-through'}),
            ('font', 'span', {}),
        ]
        for old_tag, new_tag, attrs in conversions:
            for ns in [f'{{{XHTML_NS}}}', '']:
                for elem in root.findall(f'.//{ns}{old_tag}'):
                    elem.tag = XHTML(new_tag) if ns else new_tag
                    for k, v in attrs.items():
                        existing = elem.get(k, '')
                        elem.set(k, f'{existing}; {v}' if existing else v)
        
        # Remove empty <a>, <i>, <b>, <u>, <span> tags (no content, no id/name)
        for tag in ['a', 'i', 'b', 'u', 'span', 'em', 'strong']:
            for ns in [f'{{{XHTML_NS}}}', '']:
                for elem in root.findall(f'.//{ns}{tag}'):
                    if (elem.get('id') is None and elem.get('name') is None and
                        len(elem) == 0 and not (elem.text and elem.text.strip())):
                        self._remove_element_keep_tail(elem)
                        self.stats['empty_tags_removed'] += 1
        
        # Convert <br> with content to <div>
        for ns in [f'{{{XHTML_NS}}}', '']:
            for br in root.findall(f'.//{ns}br'):
                if len(br) > 0 or (br.text and br.text.strip()):
                    br.tag = XHTML('div')
        
        # Clean text content
        for elem in root.iter():
            if elem.text:
                elem.text = self.clean_text(elem.text)
            if elem.tail:
                elem.tail = self.clean_text(elem.tail)
        
        # Fix duplicate IDs
        seen_ids = set()
        for elem in root.iter():
            id_val = elem.get('id')
            if id_val:
                if id_val in seen_ids:
                    del elem.attrib['id']
                else:
                    seen_ids.add(id_val)
        
        return root
    
    def _remove_element_keep_tail(self, elem):
        """Remove element but keep its tail text."""
        parent = elem.getparent()
        if parent is None:
            return
        
        idx = list(parent).index(elem)
        if elem.tail:
            if idx > 0:
                prev = parent[idx - 1]
                prev.tail = (prev.tail or '') + elem.tail
            else:
                parent.text = (parent.text or '') + elem.tail
        parent.remove(elem)
    
    def serialize(self, root: etree._Element) -> bytes:
        """Serialize element tree to bytes with proper EPUB formatting."""
        
        # Fix comments with -- (trips up Adobe Digital Editions)
        for comment in root.iter(etree.Comment):
            if comment.text and '--' in comment.text:
                comment.text = comment.text.replace('--', '__')
        
        # Serialize to bytes
        result = etree.tostring(root, encoding='utf-8', xml_declaration=True, pretty_print=True)
        
        # Fix self-closing tags that shouldn't be self-closing
        result = self._fix_self_closing_tags(result)
        
        return result
    
    def _fix_self_closing_tags(self, data: bytes) -> bytes:
        """Convert self-closing tags to properly closed tags for browser compatibility."""
        # Pattern to match self-closing tags like <div/> or <div />
        pattern_str = r'<({})(\s[^>]*)?\s*/>'.format('|'.join(SELF_CLOSING_BAD_TAGS))
        pattern = re.compile(pattern_str.encode('utf-8'), re.IGNORECASE)
        
        def replace_func(match):
            tag = match.group(1)
            attrs = match.group(2) or b''
            return b'<' + tag + attrs + b'></' + tag + b'>'
        
        result = pattern.sub(replace_func, data)
        self.stats['self_closing_fixed'] += len(pattern.findall(data))
        return result
    
    def process(self, data: bytes, filename: str = '<string>') -> Optional[bytes]:
        """Full processing pipeline for an XHTML file."""
        root = self.parse_xhtml(data, filename)
        if root is None:
            return None
        
        root = self.fix_structure(root)
        root = self.clean_content(root)
        return self.serialize(root)


# ============================================================================
# EPUB PROCESSOR
# ============================================================================
class EPUBProcessor:
    """Process EPUB files - fix structure and optionally translate."""
    
    def __init__(self, translate=False, max_workers=0, request_interval=0.0,
                 custom_watermarks=None, verbose=True):
        self.translate = translate
        self.verbose = verbose
        
        watermarks = DEFAULT_WATERMARKS + (custom_watermarks or [])
        self.xhtml_processor = XHTMLProcessor(watermarks, verbose)
        
        self.translator = None
        if translate:
            self.translator = GoogleFreeTranslate(
                max_workers=max_workers,
                request_interval=request_interval,
                verbose=verbose
            )
        
        self.stats = {'files_processed': 0}
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    
    def process(self, input_path: str, output_path: str) -> dict:
        """Process an EPUB file."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.log("Extracting EPUB...")
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(temp_dir)
            
            # Find content files
            xhtml_files = []
            skip_patterns = ['toc', 'nav', 'cover', 'copyright', 'title']
            
            for root, dirs, files in os.walk(temp_dir):
                for filename in files:
                    ext = Path(filename).suffix.lower()
                    if ext not in {'.xhtml', '.html', '.htm', '.xml'}:
                        continue
                    
                    lower_name = filename.lower()
                    if any(skip in lower_name for skip in skip_patterns):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    xhtml_files.append((filepath, filename))
            
            xhtml_files.sort(key=lambda x: x[1])
            total_files = len(xhtml_files)
            self.log(f"Found {total_files} content files to process")
            
            # Phase 1: Process and collect Chinese text
            all_texts = []  # (file_idx, xpath, text)
            processed_roots = []  # Store parsed trees
            file_paths = []
            
            for idx, (filepath, filename) in enumerate(xhtml_files):
                with open(filepath, 'rb') as f:
                    data = f.read()
                
                root = self.xhtml_processor.parse_xhtml(data, filename)
                if root is None:
                    self.log(f"  [{idx+1}/{total_files}] ✗ {filename} (parse error)")
                    processed_roots.append(None)
                    file_paths.append(None)
                    continue
                
                root = self.xhtml_processor.fix_structure(root)
                root = self.xhtml_processor.clean_content(root)
                processed_roots.append(root)
                file_paths.append(filepath)
                
                # Collect Chinese text for translation
                if self.translate:
                    for elem in root.iter():
                        if elem.text and self.is_chinese(elem.text):
                            all_texts.append((idx, 'text', id(elem), elem.text.strip()))
                        if elem.tail and self.is_chinese(elem.tail):
                            all_texts.append((idx, 'tail', id(elem), elem.tail.strip()))
                
                self.stats['files_processed'] += 1
            
            self.log(f"  Found {len(all_texts)} text segments to translate")
            
            # Phase 2: Translate
            if self.translate and self.translator and all_texts:
                self.log("\nTranslating...")
                texts_to_translate = [t[3] for t in all_texts]
                translated_texts = self.translator.translate_concurrent(texts_to_translate)
                
                # Build lookup for translations
                translations = {}
                for i, (file_idx, attr, elem_id, original) in enumerate(all_texts):
                    translations[(file_idx, attr, elem_id)] = translated_texts[i]
                
                # Apply translations
                for idx, root in enumerate(processed_roots):
                    if root is None:
                        continue
                    for elem in root.iter():
                        key_text = (idx, 'text', id(elem))
                        key_tail = (idx, 'tail', id(elem))
                        if key_text in translations:
                            # Preserve leading/trailing whitespace
                            orig = elem.text or ''
                            leading = len(orig) - len(orig.lstrip())
                            trailing = len(orig) - len(orig.rstrip())
                            trans = translations[key_text]
                            elem.text = orig[:leading] + trans + (orig[-trailing:] if trailing else '')
                        if key_tail in translations:
                            orig = elem.tail or ''
                            leading = len(orig) - len(orig.lstrip())
                            trailing = len(orig) - len(orig.rstrip())
                            trans = translations[key_tail]
                            elem.tail = orig[:leading] + trans + (orig[-trailing:] if trailing else '')
            
            # Phase 3: Write files
            self.log("\nWriting files...")
            for idx, (filepath, filename) in enumerate(xhtml_files):
                root = processed_roots[idx]
                if root is None:
                    continue
                
                serialized = self.xhtml_processor.serialize(root)
                with open(filepath, 'wb') as f:
                    f.write(serialized)
                
                self.log(f"  [{idx+1}/{total_files}] ✓ {filename}")
            
            # Repackage EPUB
            self.log("\nRepackaging EPUB...")
            self._create_epub(temp_dir, output_path)
        
        # Merge stats
        self.stats.update(self.xhtml_processor.stats)
        if self.translator:
            self.stats.update(self.translator.stats)
        
        return self.stats
    
    def _create_epub(self, source_dir: str, output_path: str):
        """Create a valid EPUB file from directory."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Write mimetype first, uncompressed (EPUB requirement)
            mimetype_path = os.path.join(source_dir, 'mimetype')
            if os.path.exists(mimetype_path):
                zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
            else:
                zf.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
            
            # Write everything else
            exclude = {'.DS_Store', 'mimetype', 'iTunesMetadata.plist', 'Thumbs.db'}
            for root, dirs, files in os.walk(source_dir):
                for filename in files:
                    if filename in exclude:
                        continue
                    filepath = os.path.join(root, filename)
                    # Normalize path to NFC (Unicode normalization)
                    arcname = unicodedata.normalize('NFC', 
                        os.path.relpath(filepath, source_dir).replace(os.sep, '/'))
                    zf.write(filepath, arcname)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='EPUB Fixer & Translator - Fix and translate Chinese EPUBs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s novel.epub                     # Fix and translate (default)
  %(prog)s novel.epub --no-translate      # Fix only  
  %(prog)s novel.epub -o out.epub         # Custom output name
  %(prog)s novel.epub --workers 50        # Limit concurrent requests
        """
    )
    
    parser.add_argument('input', help='Input EPUB file')
    parser.add_argument('-o', '--output', help='Output EPUB file')
    parser.add_argument('--no-translate', action='store_true', help='Skip translation')
    parser.add_argument('--workers', type=int, default=0,
                        help='Max concurrent translation requests (0=auto)')
    parser.add_argument('--interval', type=float, default=0.0,
                        help='Delay between requests in seconds')
    parser.add_argument('--add-watermark', action='append', default=[],
                        help='Add custom watermark pattern (regex)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    do_translate = not args.no_translate
    
    output_path = args.output
    if not output_path:
        stem = Path(args.input).stem
        output_path = f"{stem}_{'translated' if do_translate else 'fixed'}.epub"
    
    processor = EPUBProcessor(
        translate=do_translate,
        max_workers=args.workers,
        request_interval=args.interval,
        custom_watermarks=args.add_watermark,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print(f"EPUB Fixer {'& Translator' if do_translate else '(Fix Only)'}")
        print("=" * 50)
        print(f"Input:  {args.input}")
        print(f"Output: {output_path}")
        print()
    
    start_time = time.time()
    stats = processor.process(args.input, output_path)
    elapsed = time.time() - start_time
    
    if not args.quiet:
        print(f"\n{'=' * 50}")
        print("Summary:")
        print(f"  Files processed:      {stats.get('files_processed', 0)}")
        print(f"  Elements removed:     {stats.get('elements_removed', 0)}")
        print(f"  Empty tags removed:   {stats.get('empty_tags_removed', 0)}")
        print(f"  Self-closing fixed:   {stats.get('self_closing_fixed', 0)}")
        print(f"  Text cleaned:         {stats.get('chars_cleaned', 0)}")
        
        if do_translate:
            print(f"\nTranslation:")
            print(f"  API requests:         {stats.get('requests', 0)}")
            print(f"  Paragraphs:           {stats.get('paragraphs_translated', 0)}")
            print(f"  Characters:           {stats.get('characters_translated', 0)}")
            print(f"  Cache hits:           {stats.get('cache_hits', 0)}")
            print(f"  Errors:               {stats.get('errors', 0)}")
        
        print(f"\nCompleted in {elapsed:.1f} seconds")
        print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
