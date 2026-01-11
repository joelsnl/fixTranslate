# EPUB Fixer & Translator

A Python tool to fix, translate, and process Chinese EPUBs for reading on Google Play Books (or any e-reader).

**One command does it all:**
```bash
python epub_fixer.py novel.epub
```

## What It Does

1. **Fixes structural issues** from WebToEpub and similar tools:
   - Wraps loose text in proper `<p>` tags
   - Converts `<br>` tags to proper paragraphs
   - Removes empty spacer elements
   - Strips watermarks and ads from Chinese novel sites
   - Removes invisible/zero-width characters

2. **Translates Chinese → English** using Google Translate (Free):
   - **Concurrent translation** - up to 100 parallel requests (same speed as Calibre plugin!)
   - Same API implementation as the [Calibre Ebook Translator plugin](https://github.com/bookfere/Ebook-Translator-Calibre-Plugin)
   - Automatic retry with backoff on failures
   - In-memory caching to avoid duplicate requests

3. **Runs Calibre heuristic processing** for compatibility:
   - Fixes EPUB validation issues
   - Ensures compatibility with Google Play Books and other strict readers

## Requirements

- Python 3.7+
- [Calibre](https://calibre-ebook.com/) (for the final processing step)
- `requests` library

## Installation

1. Install Calibre from https://calibre-ebook.com/

2. Install the required Python package:
   ```bash
   pip install requests
   ```

3. Download `epub_fixer.py` and you're ready to go.

## Usage

### Basic Usage (Recommended)

```bash
python epub_fixer.py novel.epub
```

This runs the full pipeline: fix → translate → Calibre process. Output is saved as `novel_translated.epub`.

### Custom Output Name

```bash
python epub_fixer.py novel.epub -o "My Novel English.epub"
```

### Skip Steps

```bash
# Fix and translate only (skip Calibre processing)
python epub_fixer.py novel.epub --no-calibre

# Fix and Calibre process only (skip translation)
python epub_fixer.py novel.epub --no-translate
```

### Control Concurrency

```bash
# Limit to 50 concurrent translation requests
python epub_fixer.py novel.epub --workers 50

# Add delay between requests (if getting rate limited)
python epub_fixer.py novel.epub --interval 0.1
```

### Quiet Mode

```bash
python epub_fixer.py novel.epub -q
```

## All Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Custom output filename |
| `--no-translate` | Skip translation |
| `--no-calibre` | Skip Calibre heuristic processing |
| `--workers N` | Max concurrent translation requests (0=auto, default: 0) |
| `--interval SECONDS` | Delay between translation requests (default: 0) |
| `--no-wrap-text` | Don't wrap loose text in `<p>` tags |
| `--no-convert-br` | Don't convert `<br>` to paragraphs |
| `--no-remove-empty` | Keep empty elements |
| `--no-watermarks` | Keep watermarks |
| `--no-invisible` | Keep invisible characters |
| `--add-watermark PATTERN` | Add custom watermark regex pattern |
| `-q, --quiet` | Suppress output except errors |

## Watermark Removal

The script automatically removes common Chinese novel site watermarks including:
- 本書由...首發
- 請到...閱讀
- 最新章節...
- 百度搜索...
- 關注公眾號...
- And many more

To add custom patterns:
```bash
python epub_fixer.py novel.epub --add-watermark "我的自定义水印.*"
```

## How It Works

### Translation

Uses the same Google Translate Free API as the Calibre Ebook Translator plugin:
- Endpoint: `translate.googleapis.com/translate_a/single`
- GET requests for text ≤1800 chars, POST for longer
- **Concurrent requests** using `ThreadPoolExecutor` (up to 100 workers by default)
- Thread-safe caching to avoid duplicate translations
- Parses the `sentences` array from the JSON response

This concurrent approach matches the Calibre plugin's `asyncio` handler, giving you the same ~40x speed improvement over sequential translation.

### EPUB Processing

1. Extracts EPUB to temp directory
2. Processes each XHTML file:
   - Only modifies content inside `<body>` tags
   - Preserves all document structure, namespaces, and metadata
3. Repackages with proper EPUB structure (mimetype first, uncompressed)

### Calibre Integration

Automatically finds and runs `ebook-convert` with `--enable-heuristics` flag. Searches common installation paths on Windows, macOS, and Linux.

## Workflow Example

1. Use [WebToEpub](https://github.com/nickvergessen/WebToEpub) browser extension to download a Chinese web novel
2. Run: `python epub_fixer.py downloaded_novel.epub`
3. Upload `downloaded_novel_translated.epub` to Google Play Books
4. Read!

## Troubleshooting

### "ebook-convert not found"

Make sure Calibre is installed. If installed in a non-standard location, add it to your PATH.

### Translation errors / rate limiting

The script uses up to 100 concurrent requests by default. If you're getting errors:
- Reduce workers: `--workers 20`
- Add delay: `--interval 0.1`
- Check your internet connection

### Google Play Books "Processing failed"

This is why the Calibre step exists. Make sure you're not using `--no-calibre`.

## Credits

- Translation API implementation based on [Ebook Translator Calibre Plugin](https://github.com/bookfere/Ebook-Translator-Calibre-Plugin) by bookfere
- Uses [Calibre](https://calibre-ebook.com/) for final EPUB processing

## License

MIT License - Use freely, modify as needed.
