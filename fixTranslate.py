#!/usr/bin/env python3
"""
EPUB Fixer & Translator - Fix and translate Chinese EPUBs in one step

Fixes common structural issues from WebToEpub:
- Loose text not wrapped in paragraph tags
- <br> tags used instead of proper paragraphs
- Empty spacer elements
- Watermarks and ads from novel sites
- Invisible characters

Then translates Chinese to English using Google Translate (Free).
Finally runs Calibre heuristic processing for Google Play Books compatibility.

Translation implementation matches the Calibre Ebook Translator plugin exactly.

Usage: 
    python epub_fixer.py input.epub                  # Full pipeline (fix + translate + calibre)
    python epub_fixer.py input.epub --no-translate   # Fix and calibre only
    python epub_fixer.py input.epub --no-calibre     # Fix and translate only
"""

import sys
import zipfile
import tempfile
import os
import re
import argparse
import time
import json
import subprocess
import shutil
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict


# ============================================================================
# WATERMARK PATTERNS - Common watermarks from Chinese novel sites
# ============================================================================
DEFAULT_WATERMARKS = [
    r'本書由.{0,30}首發',
    r'本文由.{0,30}首發',
    r'正版請.{0,30}閱讀',
    r'請到.{0,30}閱讀',
    r'最新章節.{0,30}閱讀',
    r'手機閱讀.{0,50}',
    r'訪問下載.{0,50}',
    r'更多精彩.{0,50}',
    r'歡迎廣大書友.{0,50}',
    r'喜歡請收藏.{0,50}',
    r'請記住本書.{0,50}',
    r'百度搜索.{0,50}',
    r'最快更新.{0,50}',
    r'無彈窗.{0,30}',
    r'[𝕒-𝕫𝔸-𝕫𝟘-𝟡]+\.[𝕒-𝕫𝔸-𝕫]+',
    r'[ａ-ｚＡ-Ｚ０-９]+\.[ａ-ｚＡ-Ｚ]+',
    r'關注公眾號.{0,50}',
    r'微信公眾號.{0,50}',
    r'掃碼關注.{0,50}',
    r'點擊下載.{0,50}',
    r'APP下載.{0,50}',
]

INVISIBLE_CHARS = [
    '\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad', '\u2060',
    '\u180e', '\u200e', '\u200f', '\u202a', '\u202b', '\u202c',
    '\u202d', '\u202e',
]


# ============================================================================
# GOOGLE TRANSLATE (FREE) - Using requests library like plugin uses mechanize
# ============================================================================
class GoogleFreeTranslate:
    """
    Google Translate Free API - matches Calibre Ebook Translator plugin.
    Uses requests library (similar to how plugin uses mechanize).
    """
    
    # Same endpoint as Calibre plugin
    ENDPOINT = 'https://translate.googleapis.com/translate_a/single'
    
    # Same User-Agent as Calibre plugin
    USER_AGENT = (
        'DeepLBrowserExtension/1.3.0 Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    )
    
    def __init__(
        self, 
        source_lang: str = 'zh-CN',
        target_lang: str = 'en',
        request_timeout: int = 10,
        request_attempt: int = 3,
        request_interval: float = 0.0,
        verbose: bool = True
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.request_timeout = request_timeout
        self.request_attempt = request_attempt
        self.request_interval = request_interval
        self.verbose = verbose
        
        self.stats = {
            'requests': 0,
            'paragraphs_translated': 0,
            'characters_translated': 0,
            'errors': 0,
        }
        
        # Cache translations
        self.cache: Dict[str, str] = {}
        
        # Session for connection pooling (like mechanize Browser)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.USER_AGENT,
        })
    
    def _get_params(self, text: str) -> dict:
        """Same params as Calibre plugin GoogleFreeTranslate.get_body()"""
        return {
            'client': 'gtx',
            'sl': self.source_lang,
            'tl': self.target_lang,
            'dt': 't',
            'dj': '1',
            'q': text,
        }
    
    def _get_result(self, data: dict) -> str:
        """Same parsing as Calibre plugin GoogleFreeTranslate.get_result()"""
        if 'sentences' in data:
            return ''.join(s.get('trans', '') for s in data['sentences'] if 'trans' in s)
        return ''
    
    def _make_request(self, text: str) -> str:
        """Make translation request - matches plugin behavior."""
        params = self._get_params(text)
        
        last_error = None
        interval = 0
        
        for attempt in range(self.request_attempt):
            try:
                # Same logic as plugin: GET for <= 1800 chars, POST for longer
                if len(text) <= 1800:
                    response = self.session.get(
                        self.ENDPOINT,
                        params=params,
                        timeout=self.request_timeout
                    )
                else:
                    response = self.session.post(
                        self.ENDPOINT,
                        data=params,
                        timeout=self.request_timeout
                    )
                
                response.raise_for_status()
                return self._get_result(response.json())
                    
            except Exception as e:
                last_error = e
                if attempt < self.request_attempt - 1:
                    interval += 5
                    if self.verbose:
                        print(f"    Retry {attempt + 1}/{self.request_attempt}, "
                              f"waiting {interval}s: {str(e)[:60]}")
                    time.sleep(interval)
        
        self.stats['errors'] += 1
        if self.verbose:
            print(f"    Translation failed after {self.request_attempt} attempts: {last_error}")
        return text
    
    def translate(self, text: str) -> str:
        """Translate a single piece of text."""
        if not text or not text.strip():
            return text
        
        cache_key = text.strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        translated = self._make_request(text)
        self.stats['requests'] += 1
        self.stats['paragraphs_translated'] += 1
        self.stats['characters_translated'] += len(text)
        
        self.cache[cache_key] = translated
        
        if self.request_interval > 0:
            time.sleep(self.request_interval)
        
        return translated
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts."""
        return [self.translate(text) for text in texts]


# ============================================================================
# EPUB FIXER CLASS
# ============================================================================
class EPUBFixer:
    """Safe EPUB fixer that only modifies body content, with optional translation."""
    
    def __init__(
        self,
        wrap_loose_text: bool = True,
        convert_br_to_p: bool = True,
        remove_empty_elements: bool = True,
        remove_watermarks: bool = True,
        remove_invisible_chars: bool = True,
        translate: bool = False,
        request_interval: float = 0.0,
        custom_watermarks: List[str] = None,
        verbose: bool = True
    ):
        self.wrap_loose_text = wrap_loose_text
        self.convert_br_to_p = convert_br_to_p
        self.remove_empty_elements = remove_empty_elements
        self.remove_watermarks = remove_watermarks
        self.remove_invisible_chars = remove_invisible_chars
        self.translate = translate
        self.verbose = verbose
        
        # Initialize translator if needed
        self.translator = None
        if translate:
            self.translator = GoogleFreeTranslate(
                request_interval=request_interval,
                verbose=verbose
            )
        
        # Compile watermark patterns
        self.watermark_patterns = []
        all_watermarks = DEFAULT_WATERMARKS + (custom_watermarks or [])
        for pattern in all_watermarks:
            try:
                self.watermark_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                if self.verbose:
                    print(f"  Warning: Invalid watermark pattern '{pattern}': {e}")
        
        # Stats
        self.stats = {
            'files_processed': 0,
            'paragraphs_wrapped': 0,
            'br_converted': 0,
            'empty_removed': 0,
            'watermarks_removed': 0,
            'invisible_removed': 0,
        }
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_pattern = re.compile(
            r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df'
            r'\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf'
            r'\uf900-\ufaff\u2f800-\u2fa1f]'
        )
        return bool(chinese_pattern.search(text))
    
    def clean_text_content(self, text: str) -> str:
        """Clean text content - remove watermarks and invisible chars."""
        original_len = len(text)
        
        if self.remove_invisible_chars:
            for char in INVISIBLE_CHARS:
                text = text.replace(char, '')
            removed = original_len - len(text)
            if removed > 0:
                self.stats['invisible_removed'] += removed
        
        if self.remove_watermarks:
            for pattern in self.watermark_patterns:
                matches = pattern.findall(text)
                if matches:
                    self.stats['watermarks_removed'] += len(matches)
                    text = pattern.sub('', text)
        
        return text
    
    def process_body_content(self, body_content: str) -> str:
        """Process body content: clean, fix structure, optionally translate."""
        
        # Step 1: Convert <br> tags to newlines
        if self.convert_br_to_p:
            br_count = len(re.findall(r'<br\s*/?>', body_content))
            self.stats['br_converted'] += br_count
            body_content = re.sub(r'<br\s*/?>\s*', '\n', body_content)
        
        # Step 2: Remove empty paragraphs and divs
        if self.remove_empty_elements:
            empty_p_pattern = r'<p[^>]*>\s*(?:&nbsp;|&#160;|\s| )*\s*</p>'
            matches = len(re.findall(empty_p_pattern, body_content))
            body_content = re.sub(empty_p_pattern, '\n', body_content)
            self.stats['empty_removed'] += matches
            
            empty_div_pattern = r'<div[^>]*>\s*(?:&nbsp;|&#160;|\s| )*\s*</div>'
            matches = len(re.findall(empty_div_pattern, body_content))
            body_content = re.sub(empty_div_pattern, '\n', body_content)
            self.stats['empty_removed'] += matches
        
        # Step 3: Process tokens
        token_pattern = r'(<[^>]+>)'
        tokens = re.split(token_pattern, body_content)
        
        result_tokens = []
        block_elements = {
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
            'p', 'div', 'blockquote', 'pre', 
            'ul', 'ol', 'li', 
            'table', 'tr', 'td', 'th', 'thead', 'tbody',
            'article', 'section', 'header', 'footer', 'nav', 'aside',
            'figure', 'figcaption', 'main', 'address',
        }
        
        tag_stack = []
        
        for token in tokens:
            if not token:
                continue
            
            if token.startswith('<'):
                tag_match = re.match(r'<(/?)(\w+)', token)
                if tag_match:
                    is_closing = tag_match.group(1) == '/'
                    tag_name = tag_match.group(2).lower()
                    
                    if tag_name in block_elements:
                        if is_closing:
                            if tag_stack and tag_stack[-1] == tag_name:
                                tag_stack.pop()
                        elif not token.rstrip().endswith('/>'):
                            tag_stack.append(tag_name)
                
                result_tokens.append(token)
            else:
                # Text content - clean it
                text = self.clean_text_content(token)
                in_block = len(tag_stack) > 0
                
                if not in_block and self.wrap_loose_text:
                    lines = text.split('\n')
                    wrapped_lines = []
                    
                    for line in lines:
                        stripped = line.strip()
                        if stripped:
                            has_content = re.search(
                                r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df'
                                r'\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf'
                                r'\uf900-\ufaff\u2f800-\u2fa1f'
                                r'\u3040-\u309f\u30a0-\u30ff'
                                r'\uac00-\ud7af'
                                r'a-zA-Z0-9]', 
                                stripped
                            )
                            if has_content:
                                wrapped_lines.append(f'<p>{stripped}</p>')
                                self.stats['paragraphs_wrapped'] += 1
                            else:
                                wrapped_lines.append(line)
                        else:
                            wrapped_lines.append('')
                    
                    result_tokens.append('\n'.join(wrapped_lines))
                else:
                    result_tokens.append(text)
        
        new_body = ''.join(result_tokens)
        new_body = re.sub(r'\n{3,}', '\n\n', new_body)
        
        # Step 4: Translate if enabled
        if self.translate and self.translator:
            new_body = self.translate_body_content(new_body)
        
        return new_body
    
    def translate_body_content(self, body_content: str) -> str:
        """Translate text content within HTML tags."""
        # Split into tags and text
        token_pattern = r'(<[^>]+>)'
        tokens = re.split(token_pattern, body_content)
        
        result_tokens = []
        
        for i, token in enumerate(tokens):
            if token and not token.startswith('<'):
                stripped = token.strip()
                if stripped and self.is_chinese(stripped):
                    # Translate this text
                    translated = self.translator.translate(stripped)
                    
                    # Preserve leading/trailing whitespace
                    leading_ws = len(token) - len(token.lstrip())
                    trailing_ws = len(token) - len(token.rstrip())
                    
                    if trailing_ws > 0:
                        result_tokens.append(
                            token[:leading_ws] + translated + token[-trailing_ws:]
                        )
                    else:
                        result_tokens.append(token[:leading_ws] + translated)
                else:
                    result_tokens.append(token)
            else:
                result_tokens.append(token)
        
        return ''.join(result_tokens)
    
    def fix_xhtml_content(self, content: str) -> str:
        """Apply fixes to XHTML content. Only modifies <body> content."""
        body_match = re.search(
            r'(<body[^>]*>)(.*?)(</body>)', 
            content, 
            re.DOTALL | re.IGNORECASE
        )
        
        if not body_match:
            return content
        
        before_body = content[:body_match.start()]
        body_open_tag = body_match.group(1)
        body_content = body_match.group(2)
        body_close_tag = body_match.group(3)
        after_body = content[body_match.end():]
        
        fixed_body_content = self.process_body_content(body_content)
        
        return before_body + body_open_tag + fixed_body_content + body_close_tag + after_body
    
    def process_epub(self, input_path: str, output_path: str) -> dict:
        """Process an EPUB file and return statistics."""
        
        self.stats = {k: 0 for k in self.stats}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.log(f"Extracting EPUB...")
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            xhtml_extensions = {'.xhtml', '.html', '.htm'}
            skip_patterns = ['toc', 'nav', 'cover', 'copyright', 'title']
            
            # Collect files to process
            files_to_process = []
            for root, dirs, files in os.walk(temp_dir):
                for filename in files:
                    ext = Path(filename).suffix.lower()
                    if ext not in xhtml_extensions:
                        continue
                    
                    lower_name = filename.lower()
                    if any(skip in lower_name for skip in skip_patterns):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    files_to_process.append((filepath, filename))
            
            # Sort files for consistent order
            files_to_process.sort(key=lambda x: x[1])
            
            total_files = len(files_to_process)
            self.log(f"Processing {total_files} content files...")
            
            for idx, (filepath, filename) in enumerate(files_to_process, 1):
                try:
                    content = None
                    for encoding in ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1']:
                        try:
                            with open(filepath, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        self.log(f"  [{idx}/{total_files}] ✗ {filename} (decode error)")
                        continue
                    
                    if '<body' not in content.lower():
                        continue
                    
                    fixed_content = self.fix_xhtml_content(content)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    self.stats['files_processed'] += 1
                    self.log(f"  [{idx}/{total_files}] ✓ {filename}")
                
                except Exception as e:
                    self.log(f"  [{idx}/{total_files}] ✗ {filename}: {e}")
            
            # Repackage
            self.log(f"\nRepackaging EPUB...")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                mimetype_path = os.path.join(temp_dir, 'mimetype')
                if os.path.exists(mimetype_path):
                    zipf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
                
                for root, dirs, files in os.walk(temp_dir):
                    for filename in files:
                        if filename == 'mimetype':
                            continue
                        filepath = os.path.join(root, filename)
                        arcname = os.path.relpath(filepath, temp_dir)
                        zipf.write(filepath, arcname)
        
        # Merge translator stats if available
        if self.translator:
            self.stats.update(self.translator.stats)
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description='EPUB Fixer & Translator - Fix and translate Chinese EPUBs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s novel.epub                     # Fix, translate, and Calibre process (default)
  %(prog)s novel.epub --no-calibre        # Fix and translate only
  %(prog)s novel.epub --no-translate      # Fix and Calibre process only
  %(prog)s novel.epub -o out.epub         # Custom output name
  %(prog)s novel.epub --interval 0.5      # Add delay between requests
  %(prog)s novel.epub --no-watermarks     # Keep watermarks
        """
    )
    
    parser.add_argument('input', help='Input EPUB file')
    parser.add_argument('-o', '--output', help='Output EPUB file')
    
    # Translation (on by default)
    parser.add_argument('--no-translate', action='store_true',
                        help='Skip translation (fix structure only)')
    parser.add_argument('--interval', type=float, default=0.0,
                        help='Delay between translation requests in seconds (default: 0)')
    
    # Calibre post-processing (on by default)
    parser.add_argument('--no-calibre', action='store_true',
                        help='Skip Calibre heuristic processing')
    
    # Fix toggles
    parser.add_argument('--no-wrap-text', action='store_true',
                        help='Disable wrapping loose text in <p> tags')
    parser.add_argument('--no-convert-br', action='store_true',
                        help='Disable converting <br> to paragraphs')
    parser.add_argument('--no-remove-empty', action='store_true',
                        help='Disable removing empty elements')
    parser.add_argument('--no-watermarks', action='store_true',
                        help='Disable watermark removal')
    parser.add_argument('--no-invisible', action='store_true',
                        help='Disable invisible character removal')
    
    parser.add_argument('--add-watermark', action='append', default=[],
                        help='Add custom watermark pattern (regex)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output except errors')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    # Derive flags (translate and calibre are ON by default)
    do_translate = not args.no_translate
    do_calibre = not args.no_calibre
    
    # Generate output filename
    output_path = args.output
    if not output_path:
        input_stem = Path(args.input).stem
        if do_translate:
            output_path = f"{input_stem}_translated.epub"
        else:
            output_path = f"{input_stem}_fixed.epub"
    
    fixer = EPUBFixer(
        wrap_loose_text=not args.no_wrap_text,
        convert_br_to_p=not args.no_convert_br,
        remove_empty_elements=not args.no_remove_empty,
        remove_watermarks=not args.no_watermarks,
        remove_invisible_chars=not args.no_invisible,
        translate=do_translate,
        request_interval=args.interval,
        custom_watermarks=args.add_watermark,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        if do_translate and do_calibre:
            mode = "Fix, Translate & Calibre"
        elif do_translate:
            mode = "Fix & Translate"
        elif do_calibre:
            mode = "Fix & Calibre"
        else:
            mode = "Fix Only"
        print(f"EPUB Fixer ({mode})")
        print(f"=" * 50)
        print(f"Input:  {args.input}")
        print(f"Output: {output_path}")
        print()
    
    start_time = time.time()
    stats = fixer.process_epub(args.input, output_path)
    elapsed = time.time() - start_time
    
    if not args.quiet:
        print(f"\n{'=' * 50}")
        print(f"Summary:")
        print(f"  Files processed:     {stats['files_processed']}")
        print(f"  Paragraphs wrapped:  {stats['paragraphs_wrapped']}")
        print(f"  <br> tags converted: {stats['br_converted']}")
        print(f"  Empty elements:      {stats['empty_removed']}")
        print(f"  Watermarks removed:  {stats['watermarks_removed']}")
        print(f"  Invisible chars:     {stats['invisible_removed']}")
        
        if do_translate:
            print(f"\nTranslation:")
            print(f"  API requests:          {stats.get('requests', 0)}")
            print(f"  Paragraphs translated: {stats.get('paragraphs_translated', 0)}")
            print(f"  Characters translated: {stats.get('characters_translated', 0)}")
            print(f"  Errors:                {stats.get('errors', 0)}")
        
        print(f"\nCompleted in {elapsed:.1f} seconds")
        print(f"✓ Saved to: {output_path}")
    
    # Run Calibre ebook-convert if requested
    if do_calibre:
        if not args.quiet:
            print(f"\nRunning Calibre heuristic processing...")
        
        # Find ebook-convert
        ebook_convert = shutil.which('ebook-convert')
        if not ebook_convert:
            # Try common locations
            if sys.platform == 'win32':
                possible_paths = [
                    r'C:\Program Files\Calibre2\ebook-convert.exe',
                    r'C:\Program Files (x86)\Calibre2\ebook-convert.exe',
                    os.path.expanduser(r'~\AppData\Local\Calibre2\ebook-convert.exe'),
                ]
            elif sys.platform == 'darwin':
                possible_paths = [
                    '/Applications/calibre.app/Contents/MacOS/ebook-convert',
                ]
            else:
                possible_paths = [
                    '/usr/bin/ebook-convert',
                    '/usr/local/bin/ebook-convert',
                ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ebook_convert = path
                    break
        
        if not ebook_convert:
            print("Error: ebook-convert not found. Please ensure Calibre is installed.")
            print("       You can manually run: ebook-convert output.epub output.epub --enable-heuristics")
            sys.exit(1)
        
        # Create temp file for conversion
        temp_output = output_path + '.tmp.epub'
        
        try:
            cmd = [
                ebook_convert,
                output_path,
                temp_output,
                '--enable-heuristics',
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                # Replace original with converted
                os.replace(temp_output, output_path)
                if not args.quiet:
                    print(f"✓ Calibre heuristic processing complete")
            else:
                print(f"Calibre conversion failed: {result.stderr}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                sys.exit(1)
                
        except Exception as e:
            print(f"Error running ebook-convert: {e}")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            sys.exit(1)


if __name__ == "__main__":
    main()
