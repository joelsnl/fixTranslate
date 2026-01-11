#!/usr/bin/env python3
"""
EPUB Fixer & Translator - Fix and translate Chinese EPUBs in one step

Fixes common structural issues from WebToEpub:
- Loose text not wrapped in paragraph tags
- <br> tags used instead of proper paragraphs
- Empty spacer elements
- Watermarks and ads from novel sites
- Invisible characters

EPUB validation/repair (replaces Calibre heuristics):
- Removes invalid elements (scripts, forms, embed, object)
- Fixes deprecated tags (<center>, <u>)
- Removes duplicate IDs
- Cleans special characters for e-reader compatibility
- Validates XHTML structure

Translates Chinese to English using Google Translate (Free).

Uses concurrent translation like the Calibre Ebook Translator plugin for speed.

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
import json
import subprocess
import shutil
import requests
import concurrent.futures
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from html.parser import HTMLParser
from html import unescape
import xml.etree.ElementTree as ET


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

# Characters to remove (invisible/zero-width)
INVISIBLE_CHARS = [
    '\u200b',  # Zero-width space
    '\u200c',  # Zero-width non-joiner
    '\u200d',  # Zero-width joiner
    '\ufeff',  # BOM / Zero-width no-break space
    '\u00ad',  # Soft hyphen (causes rendering issues)
    '\u2060',  # Word joiner
    '\u180e',  # Mongolian vowel separator
    '\u200e',  # Left-to-right mark
    '\u200f',  # Right-to-left mark
    '\u202a',  # Left-to-right embedding
    '\u202b',  # Right-to-left embedding
    '\u202c',  # Pop directional formatting
    '\u202d',  # Left-to-right override
    '\u202e',  # Right-to-left override
]

# Characters to replace (for e-reader compatibility)
CHAR_REPLACEMENTS = {
    '\u2011': '-',   # Non-breaking hyphen → regular hyphen
    '\u2013': '-',   # En dash → hyphen (optional, some readers handle this)
    '\u2014': '—',   # Em dash (keep as is, widely supported)
}


# ============================================================================
# GOOGLE TRANSLATE (FREE) - Concurrent implementation like Calibre plugin
# ============================================================================
class GoogleFreeTranslate:
    """
    Google Translate Free API with concurrent requests.
    Matches Calibre Ebook Translator plugin performance.
    """
    
    ENDPOINT = 'https://translate.googleapis.com/translate_a/single'
    
    USER_AGENT = (
        'DeepLBrowserExtension/1.3.0 Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    )
    
    def __init__(
        self, 
        source_lang: str = 'zh-CN',
        target_lang: str = 'en',
        max_workers: int = 0,  # 0 = auto (based on paragraph count, like plugin)
        request_timeout: int = 10,
        request_attempt: int = 3,
        request_interval: float = 0.0,
        verbose: bool = True
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.request_attempt = request_attempt
        self.request_interval = request_interval
        self.verbose = verbose
        
        self.stats = {
            'requests': 0,
            'paragraphs_translated': 0,
            'characters_translated': 0,
            'cache_hits': 0,
            'errors': 0,
        }
        
        # Thread-safe cache
        self.cache: Dict[str, str] = {}
        self.cache_lock = threading.Lock()
        
        # Thread-safe stats
        self.stats_lock = threading.Lock()
        
        # Progress tracking
        self.completed = 0
        self.total = 0
        self.progress_lock = threading.Lock()
    
    def _get_params(self, text: str) -> dict:
        return {
            'client': 'gtx',
            'sl': self.source_lang,
            'tl': self.target_lang,
            'dt': 't',
            'dj': '1',
            'q': text,
        }
    
    def _get_result(self, data: dict) -> str:
        if 'sentences' in data:
            return ''.join(s.get('trans', '') for s in data['sentences'] if 'trans' in s)
        return ''
    
    def _translate_single(self, text: str, index: int) -> Tuple[int, str]:
        """Translate a single text. Returns (index, translated_text)."""
        if not text or not text.strip():
            return (index, text)
        
        cache_key = text.strip()
        
        # Check cache (thread-safe)
        with self.cache_lock:
            if cache_key in self.cache:
                with self.stats_lock:
                    self.stats['cache_hits'] += 1
                self._update_progress()
                return (index, self.cache[cache_key])
        
        # Make request with retries
        params = self._get_params(text)
        last_error = None
        interval = 0
        
        for attempt in range(self.request_attempt):
            try:
                # GET for short, POST for long (like plugin)
                if len(text) <= 1800:
                    response = requests.get(
                        self.ENDPOINT,
                        params=params,
                        headers={'User-Agent': self.USER_AGENT},
                        timeout=self.request_timeout
                    )
                else:
                    response = requests.post(
                        self.ENDPOINT,
                        data=params,
                        headers={'User-Agent': self.USER_AGENT},
                        timeout=self.request_timeout
                    )
                
                response.raise_for_status()
                translated = self._get_result(response.json())
                
                # Cache result (thread-safe)
                with self.cache_lock:
                    self.cache[cache_key] = translated
                
                # Update stats (thread-safe)
                with self.stats_lock:
                    self.stats['requests'] += 1
                    self.stats['paragraphs_translated'] += 1
                    self.stats['characters_translated'] += len(text)
                
                self._update_progress()
                
                # Optional delay between requests
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
                
                return (index, translated)
                    
            except Exception as e:
                last_error = e
                if attempt < self.request_attempt - 1:
                    interval += 5
                    time.sleep(interval)
        
        # All retries failed
        with self.stats_lock:
            self.stats['errors'] += 1
        self._update_progress()
        return (index, text)  # Return original on failure
    
    def _update_progress(self):
        """Update and print progress."""
        with self.progress_lock:
            self.completed += 1
            if self.verbose and self.total > 0:
                pct = (self.completed / self.total) * 100
                print(f"\r  Translating: {self.completed}/{self.total} ({pct:.1f}%)", end='', flush=True)
    
    def translate_concurrent(self, texts: List[str]) -> List[str]:
        """
        Translate all texts concurrently using ThreadPoolExecutor.
        This is the key to matching Calibre plugin speed.
        """
        if not texts:
            return []
        
        self.total = len(texts)
        self.completed = 0
        
        # Determine worker count (like plugin: 0 means use all)
        workers = self.max_workers if self.max_workers > 0 else min(len(texts), 100)
        
        if self.verbose:
            print(f"  Starting translation with {workers} concurrent workers...")
        
        # Results array (same order as input)
        results = [''] * len(texts)
        
        # Submit all tasks to thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Create futures with their indices
            futures = {
                executor.submit(self._translate_single, text, i): i 
                for i, text in enumerate(texts)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, translated = future.result()
                    results[index] = translated
                except Exception as e:
                    # On exception, keep original text
                    index = futures[future]
                    results[index] = texts[index]
        
        if self.verbose:
            print()  # Newline after progress
        
        return results


# ============================================================================
# EPUB VALIDATION/REPAIR - Replaces Calibre heuristics
# ============================================================================
class EPUBValidator:
    """
    EPUB validation and repair. Performs the same fixes as Calibre's
    heuristic processing and ADE quirks workarounds.
    """
    
    # Elements to remove entirely
    REMOVE_ELEMENTS = {'script', 'embed', 'object', 'form', 'input', 'button', 'textarea'}
    
    # Elements to convert
    CONVERT_ELEMENTS = {
        'center': ('div', {'style': 'text-align:center'}),
        'u': ('span', {'style': 'text-decoration:underline'}),
        'font': ('span', {}),
        's': ('span', {'style': 'text-decoration:line-through'}),
        'strike': ('span', {'style': 'text-decoration:line-through'}),
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stats = {
            'elements_removed': 0,
            'elements_converted': 0,
            'duplicate_ids_fixed': 0,
            'special_chars_fixed': 0,
        }
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def clean_special_chars(self, text: str) -> str:
        """Remove/replace special characters for e-reader compatibility."""
        original = text
        
        # Remove invisible characters
        for char in INVISIBLE_CHARS:
            text = text.replace(char, '')
        
        # Replace problematic characters
        for old, new in CHAR_REPLACEMENTS.items():
            text = text.replace(old, new)
        
        if text != original:
            self.stats['special_chars_fixed'] += 1
        
        return text
    
    def fix_xhtml(self, content: str) -> str:
        """
        Fix XHTML content for e-reader compatibility.
        This replaces Calibre's heuristic processing.
        """
        # Track seen IDs to remove duplicates
        seen_ids = set()
        
        def process_tag(match):
            """Process a single tag, fixing issues."""
            full_tag = match.group(0)
            tag_name_match = re.match(r'<(/?)(\w+)', full_tag)
            if not tag_name_match:
                return full_tag
            
            is_closing = tag_name_match.group(1) == '/'
            tag_name = tag_name_match.group(2).lower()
            
            # Remove forbidden elements
            if tag_name in self.REMOVE_ELEMENTS:
                self.stats['elements_removed'] += 1
                return ''
            
            # Convert deprecated elements
            if tag_name in self.CONVERT_ELEMENTS and not is_closing:
                new_tag, attrs = self.CONVERT_ELEMENTS[tag_name]
                self.stats['elements_converted'] += 1
                
                # Extract existing attributes
                attr_str = full_tag[len(tag_name_match.group(0)):-1].strip()
                if attr_str.endswith('/'):
                    attr_str = attr_str[:-1].strip()
                    self_closing = True
                else:
                    self_closing = False
                
                # Merge style attributes if both exist
                existing_style = ''
                style_match = re.search(r'style\s*=\s*["\']([^"\']*)["\']', attr_str, re.IGNORECASE)
                if style_match:
                    existing_style = style_match.group(1)
                    attr_str = re.sub(r'\s*style\s*=\s*["\'][^"\']*["\']', '', attr_str, flags=re.IGNORECASE)
                
                # Build new tag
                new_attrs = []
                if attr_str.strip():
                    new_attrs.append(attr_str.strip())
                
                if 'style' in attrs:
                    combined_style = f"{existing_style}; {attrs['style']}" if existing_style else attrs['style']
                    new_attrs.append(f'style="{combined_style}"')
                
                attr_part = ' '.join(new_attrs)
                if attr_part:
                    attr_part = ' ' + attr_part
                
                if self_closing:
                    return f'<{new_tag}{attr_part}/>'
                else:
                    return f'<{new_tag}{attr_part}>'
            
            # Handle closing tags for converted elements
            if is_closing and tag_name in self.CONVERT_ELEMENTS:
                new_tag, _ = self.CONVERT_ELEMENTS[tag_name]
                return f'</{new_tag}>'
            
            # Fix duplicate IDs
            if not is_closing:
                id_match = re.search(r'\sid\s*=\s*["\']([^"\']+)["\']', full_tag, re.IGNORECASE)
                if id_match:
                    id_val = id_match.group(1)
                    if id_val in seen_ids:
                        # Remove duplicate ID
                        self.stats['duplicate_ids_fixed'] += 1
                        full_tag = re.sub(r'\s+id\s*=\s*["\'][^"\']*["\']', '', full_tag, flags=re.IGNORECASE)
                    else:
                        seen_ids.add(id_val)
            
            return full_tag
        
        # Process all tags
        tag_pattern = r'<[^>]+>'
        content = re.sub(tag_pattern, process_tag, content)
        
        # Clean special characters in text (not in tags)
        def clean_text_nodes(match):
            tag = match.group(0)
            return tag
        
        # Split by tags and clean text portions
        parts = re.split(r'(<[^>]+>)', content)
        cleaned_parts = []
        for part in parts:
            if part.startswith('<'):
                cleaned_parts.append(part)
            else:
                cleaned_parts.append(self.clean_special_chars(part))
        
        content = ''.join(cleaned_parts)
        
        # Remove empty script/style tags
        content = re.sub(r'<(script|style)[^>]*>\s*</\1>', '', content, flags=re.IGNORECASE)
        
        # Remove comments (optional, but can cause issues)
        # content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        return content


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
        validate_epub: bool = True,
        max_workers: int = 0,
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
        self.validate_epub = validate_epub
        self.verbose = verbose
        
        # Initialize translator if needed
        self.translator = None
        if translate:
            self.translator = GoogleFreeTranslate(
                max_workers=max_workers,
                request_interval=request_interval,
                verbose=verbose
            )
        
        # Initialize validator
        self.validator = EPUBValidator(verbose=verbose) if validate_epub else None
        
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
        """Process body content: clean and fix structure."""
        
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
                    
                    text = '\n'.join(wrapped_lines)
                
                result_tokens.append(text)
        
        new_body = ''.join(result_tokens)
        new_body = re.sub(r'\n{3,}', '\n\n', new_body)
        
        return new_body
    
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
            
            # PHASE 1: Fix structure and collect all Chinese text
            all_chinese_texts = []  # (file_idx, token_idx, text, leading_ws, trailing_ws, full_token)
            file_tokens = []  # Store tokenized content for each file
            file_data = []  # Store (filepath, before_body, body_open, body_close, after_body)
            
            for idx, (filepath, filename) in enumerate(files_to_process):
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
                        self.log(f"  [{idx+1}/{total_files}] ✗ {filename} (decode error)")
                        file_tokens.append(None)
                        file_data.append(None)
                        continue
                    
                    if '<body' not in content.lower():
                        file_tokens.append(None)
                        file_data.append(None)
                        continue
                    
                    # Apply EPUB validation fixes to entire content first
                    if self.validator:
                        content = self.validator.fix_xhtml(content)
                    
                    # Extract body
                    body_match = re.search(
                        r'(<body[^>]*>)(.*?)(</body>)', 
                        content, 
                        re.DOTALL | re.IGNORECASE
                    )
                    
                    if not body_match:
                        file_tokens.append(None)
                        file_data.append(None)
                        continue
                    
                    before_body = content[:body_match.start()]
                    body_open_tag = body_match.group(1)
                    body_content = body_match.group(2)
                    body_close_tag = body_match.group(3)
                    after_body = content[body_match.end():]
                    
                    # Process body content (fix structure)
                    fixed_body = self.process_body_content(body_content)
                    
                    # Tokenize for later replacement
                    token_pattern = r'(<[^>]+>)'
                    tokens = re.split(token_pattern, fixed_body)
                    file_tokens.append(tokens)
                    file_data.append((filepath, before_body, body_open_tag, body_close_tag, after_body))
                    
                    # Scan tokens for Chinese text
                    for token_idx, token in enumerate(tokens):
                        if token and not token.startswith('<'):
                            stripped = token.strip()
                            if stripped and self.is_chinese(stripped):
                                leading_ws = len(token) - len(token.lstrip())
                                trailing_ws = len(token) - len(token.rstrip())
                                all_chinese_texts.append((idx, token_idx, stripped, leading_ws, trailing_ws, token))
                    
                    self.stats['files_processed'] += 1
                    
                except Exception as e:
                    self.log(f"  [{idx+1}/{total_files}] ✗ {filename}: {e}")
                    file_tokens.append(None)
                    file_data.append(None)
            
            self.log(f"  Found {len(all_chinese_texts)} text segments to translate")
            
            # PHASE 2: Translate all Chinese text concurrently
            if self.translate and self.translator and all_chinese_texts:
                self.log(f"\nTranslating...")
                
                # Extract just the text
                texts_to_translate = [t[2] for t in all_chinese_texts]
                
                # Translate concurrently
                translated_texts = self.translator.translate_concurrent(texts_to_translate)
                
                # Map translations back to tokens
                for i, (file_idx, token_idx, original, leading_ws, trailing_ws, full_token) in enumerate(all_chinese_texts):
                    translated = translated_texts[i]
                    tokens = file_tokens[file_idx]
                    if tokens is not None:
                        # Reconstruct with original whitespace
                        if trailing_ws > 0:
                            tokens[token_idx] = full_token[:leading_ws] + translated + full_token[-trailing_ws:]
                        else:
                            tokens[token_idx] = full_token[:leading_ws] + translated
            
            # PHASE 3: Write all files
            self.log(f"\nWriting files...")
            for idx, (filepath, filename) in enumerate(files_to_process):
                if file_tokens[idx] is None or file_data[idx] is None:
                    continue
                
                tokens = file_tokens[idx]
                filepath, before_body, body_open_tag, body_close_tag, after_body = file_data[idx]
                
                # Reconstruct content
                body_content = ''.join(tokens)
                body_content = re.sub(r'\n{3,}', '\n\n', body_content)
                full_content = before_body + body_open_tag + body_content + body_close_tag + after_body
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                
                self.log(f"  [{idx+1}/{total_files}] ✓ {filename}")
            
            # Repackage
            self.log(f"\nRepackaging EPUB...")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # CRITICAL: mimetype must be first and uncompressed
                mimetype_path = os.path.join(temp_dir, 'mimetype')
                if os.path.exists(mimetype_path):
                    zipf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
                else:
                    # Create mimetype if missing
                    zipf.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
                
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
        
        # Merge validator stats if available
        if self.validator:
            self.stats.update(self.validator.stats)
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description='EPUB Fixer & Translator - Fix and translate Chinese EPUBs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s novel.epub                     # Fix, validate, and translate (default)
  %(prog)s novel.epub --no-translate      # Fix and validate only
  %(prog)s novel.epub -o out.epub         # Custom output name
  %(prog)s novel.epub --workers 50        # Limit concurrent requests
  %(prog)s novel.epub --interval 0.1      # Add delay between requests
        """
    )
    
    parser.add_argument('input', help='Input EPUB file')
    parser.add_argument('-o', '--output', help='Output EPUB file')
    
    # Translation (on by default)
    parser.add_argument('--no-translate', action='store_true',
                        help='Skip translation (fix structure only)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Max concurrent translation requests (0=auto, default: 0)')
    parser.add_argument('--interval', type=float, default=0.0,
                        help='Delay between translation requests in seconds (default: 0)')
    
    # EPUB validation (on by default)
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip EPUB validation/repair')
    
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
    
    # Derive flags (translate is ON by default)
    do_translate = not args.no_translate
    do_validate = not args.no_validate
    
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
        validate_epub=do_validate,
        max_workers=args.workers,
        request_interval=args.interval,
        custom_watermarks=args.add_watermark,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        if do_translate and do_validate:
            mode = "Fix, Validate & Translate"
        elif do_translate:
            mode = "Fix & Translate"
        elif do_validate:
            mode = "Fix & Validate"
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
        
        if do_validate:
            print(f"\nEPUB Validation:")
            print(f"  Elements removed:    {stats.get('elements_removed', 0)}")
            print(f"  Elements converted:  {stats.get('elements_converted', 0)}")
            print(f"  Duplicate IDs fixed: {stats.get('duplicate_ids_fixed', 0)}")
            print(f"  Special chars fixed: {stats.get('special_chars_fixed', 0)}")
        
        if do_translate:
            print(f"\nTranslation:")
            print(f"  API requests:          {stats.get('requests', 0)}")
            print(f"  Paragraphs translated: {stats.get('paragraphs_translated', 0)}")
            print(f"  Characters translated: {stats.get('characters_translated', 0)}")
            print(f"  Cache hits:            {stats.get('cache_hits', 0)}")
            print(f"  Errors:                {stats.get('errors', 0)}")
        
        print(f"\nCompleted in {elapsed:.1f} seconds")
        print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
