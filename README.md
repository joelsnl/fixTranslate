# EPUB Fixer & Translator

A standalone Python tool to fix, validate, and translate Chinese EPUBs for reading on Google Play Books (or any e-reader).

**No Calibre required!** All EPUB validation/repair is built-in.

**One command does it all:**
```bash
python fixTranslate.py novel.epub
```

## What It Does

1. **Fixes structural issues** from WebToEpub and similar tools:
   - Wraps loose text in proper `<p>` tags
   - Converts `<br>` tags to proper paragraphs
   - Removes empty spacer elements
   - Strips watermarks and ads from Chinese novel sites
   - Removes invisible/zero-width characters

2. **EPUB validation/repair** (replaces Calibre heuristics):
   - Removes invalid elements (scripts, forms, embed, object)
   - Converts deprecated tags (`<center>`, `<u>`, `<font>`, `<s>`, `<strike>`)
   - Fixes duplicate IDs (causes validation errors)
   - Cleans special characters for e-reader compatibility
   - Ensures proper EPUB structure (mimetype first, uncompressed)

3. **Translates Chinese → English** using Google Translate (Free):
   - **Concurrent translation** - up to 100 parallel requests (same speed as Calibre plugin!)
   - Same API implementation as the [Calibre Ebook Translator plugin](https://github.com/bookfere/Ebook-Translator-Calibre-Plugin)
   - Automatic retry with backoff on failures
   - In-memory caching to avoid duplicate requests

## Requirements

- Python 3.7+
- `requests` library

That's it! No Calibre needed.

## Installation

1. Install the required Python package:
   ```bash
   pip install requests
   ```

2. Download `fixTranslate.py` and you're ready to go.

## Usage

### Basic Usage (Recommended)

```bash
python fixTranslate.py novel.epub
```

This runs the full pipeline: fix → validate → translate. Output is saved as `novel_translated.epub`.

### Custom Output Name

```bash
python fixTranslate.py novel.epub -o "My Novel English.epub"
```

### Skip Translation (Fix Only)

```bash
python fixTranslate.py novel.epub --no-translate
```

### Control Concurrency

```bash
# Limit to 50 concurrent translation requests
python fixTranslate.py novel.epub --workers 50

# Add delay between requests (if getting rate limited)
python fixTranslate.py novel.epub --interval 0.1
```

### Quiet Mode

```bash
python fixTranslate.py novel.epub -q
```

## All Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Custom output filename |
| `--no-translate` | Skip translation |
| `--no-validate` | Skip EPUB validation/repair |
| `--workers N` | Max concurrent translation requests (0=auto, default: 0) |
| `--interval SECONDS` | Delay between translation requests in seconds (default: 0) |
| `--no-wrap-text` | Don't wrap loose text in `<p>` tags |
| `--no-convert-br` | Don't convert `<br>` to paragraphs |
| `--no-remove-empty` | Keep empty elements |
| `--no-watermarks` | Keep watermarks |
| `--no-invisible` | Keep invisible characters |
| `--add-watermark PATTERN` | Add custom watermark regex pattern |
| `-q, --quiet` | Suppress output except errors |

## How It Works

### Translation

Uses the same Google Translate Free API as the Calibre Ebook Translator plugin:
- Endpoint: `translate.googleapis.com/translate_a/single`
- GET requests for text ≤1800 chars, POST for longer
- **Concurrent requests** using `ThreadPoolExecutor` (up to 100 workers by default)
- Thread-safe caching to avoid duplicate translations
- Parses the `sentences` array from the JSON response

This concurrent approach matches the Calibre plugin's `asyncio` handler, giving you the same ~40x speed improvement over sequential translation.

### EPUB Validation

Performs the same fixes as Calibre's heuristic processing and ADE (Adobe Digital Editions) quirks workarounds:
- Removes `<script>`, `<embed>`, `<object>`, `<form>` elements
- Converts `<center>` → `<div style="text-align:center">`
- Converts `<u>` → `<span style="text-decoration:underline">`
- Removes duplicate `id` attributes (causes EPUB validation errors)
- Cleans zero-width spaces, soft hyphens, and other invisible characters
- Replaces non-breaking hyphens with regular hyphens
- Ensures `mimetype` file is first and uncompressed in the ZIP

### Structure Fixes

- Body-only modification (never touches `<head>`, DOCTYPE, or namespaces)
- Token-based parsing (safely identifies tags vs. text)
- Tag-safe cleaning (never modifies tag attributes with watermark patterns)
- Proper EPUB repackaging

## Typical Workflow

1. Use [WebToEpub](https://github.com/dteviot/WebToEpub) browser extension to download a Chinese web novel
2. Run: `python fixTranslate.py downloaded_novel.epub`
3. Upload `downloaded_novel_translated.epub` to Google Play Books
4. Read!

## Adding Custom Watermarks

If your source has site-specific watermarks:

```bash
python fixTranslate.py novel.epub --add-watermark "我的自定义水印.*"
```

## Troubleshooting

### Translation errors / rate limiting

The script uses up to 100 concurrent requests by default. If you're getting errors:
- Reduce workers: `--workers 20`
- Add delay: `--interval 0.1`
- Check your internet connection

### Google Play Books still rejects the file

Try these steps:
1. Make sure you're not using `--no-validate`
2. If the issue persists, the source EPUB may have unusual issues

### Output file is larger than expected

The script preserves all original formatting. If size is a concern, you can use Calibre's ebook-convert to optimize images and compress further.

## Credits

- **Translation API implementation** based on [Ebook Translator Calibre Plugin](https://github.com/bookfere/Ebook-Translator-Calibre-Plugin) by bookfere (GPL-3.0)
- **EPUB validation logic** based on [Calibre](https://github.com/kovidgoyal/calibre) by Kovid Goyal (GPL-3.0)
- **WebToEpub** browser extension for downloading web novels - https://github.com/dteviot/WebToEpub

### 🤖 AI-Assisted Development

This script was built with **Claude AI** (Anthropic) based on design, requirements, and ideas by **Joel**. The development process involved iterative collaboration where Joel provided the use case, tested outputs, and guided the implementation while Claude wrote the code.

## License

MIT License - Use freely, modify as needed.
