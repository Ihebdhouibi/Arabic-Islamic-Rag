---
dataset_info:
  - config_name: default
    features:
      - name: book_id
        dtype: int32
      - name: shamela_id
        dtype: int32
      - name: title_ar
        dtype: string
      - name: category_id
        dtype: int8
      - name: category_name_ar
        dtype: string
      - name: main_author_id
        dtype: int32
      - name: main_author_name_ar
        dtype: string
      - name: main_author_death_hijri
        dtype: int16
      - name: page_count
        dtype: int32
      - name: body
        dtype: string
      - name: footnotes
        dtype: string
      - name: page_num
        dtype: int16
    splits:
      - name: full
        num_bytes: 19000000000
        num_examples: 8589
    download_size: 19000000000
    dataset_size: 19000000000
  - config_name: meta
    features:
      - name: authors
        dtype: struct
      - name: book_metadata
        dtype: struct
      - name: categories
        dtype: struct
      - name: quran_verses
        dtype: struct
      - name: narrators
        dtype: struct
      - name: root_dictionary
        dtype: struct
      - name: hadith_xrefs
        dtype: struct
      - name: tafsir_xrefs
        dtype: struct
      - name: page_isnads
        dtype: struct
    splits:
      - name: meta
        num_bytes: 185000000
        num_examples: 1
configs:
  - config_name: default
    data_files:
      - split: full
        path: "*/**"
  - config_name: meta
    data_files:
      - split: meta
        path: "_meta/**"
license: mit
license_url: https://shamela.ws
language:
  - ar
multilinguality:
  - monolingual
size_categories:
  - 10M<n<100M
annotations_creators:
  - found
source_datasets:
  - original
task_categories:
  - text-generation
  - fill-mask
  - token-classification
  - question-answering
  - text-retrieval
task_ids:
  - language-modeling
  - named-entity-recognition
pretty_name: Shamela 4 — Full Islamic Library Corpus
tags:
  - islamic-studies
  - arabic
  - classical-arabic
  - quran
  - hadith
  - fiqh
  - tafsir
  - aqeedah
  - arabic-linguistics
  - shamela
---

# Shamela 4 — Full Islamic Library Corpus

A complete extraction of **al-Maktaba al-Shamela (الشاملة) v4**, containing **8,589 books** across **40 categories** of classical Islamic sciences. Extracted from the original Lucene + Sqlite Shamela DB on 2026-04-26 with ~7.6 million pages and ~19 GB of Arabic text.

## Dataset Structure

```
stage0_raw/
├── _meta/                          # Cross-cutting metadata (Parquet + JSONL)
│   ├── extraction_manifest.json    # Global extraction record
│   ├── authors.parquet             # 3,187 authors
│   ├── book_metadata.parquet       # All 8,589 book records
│   ├── categories.parquet          # 41 categories
│   ├── quran_verses.parquet        # 6,236 verses (entire Quran)
│   ├── narrators.parquet           # 18,989 hadith narrators
│   ├── root_dictionary.parquet     # ~1.95M Arabic root entries
│   ├── hadith_xrefs.parquet        # ~37K cross-references
│   ├── tafsir_xrefs.parquet        # ~84K exegesis links
│   ├── page_isnads.parquet         # ~35K transmission chains
│   └── *.jsonl                     # Text-preview equivalents
│
├── 01__العقيدة/                    # Category directories (numbered 01–40)
│   └── 1__الفواكه-العذاب-في-الرد/  # Per-book directories
│       ├── manifest.json           # Integrity checksums + metadata
│       ├── book_metadata.json      # Bibliographic record
│       ├── toc.jsonl               # Table of contents (JSONL)
│       └── pages.jsonl             # Full page text (JSONL)
│
├── ...
│
└── 40__علوم-أخرى/
```

### Per-Book Files

| File | Format | Description |
|------|--------|-------------|
| `manifest.json` | JSON | Extraction metadata, SHA-256 checksums, page/toc counts |
| `book_metadata.json` | JSON | Title, author(s), death dates, category, edition info |
| `toc.jsonl` | JSONL | Hierarchical table of contents with page anchors |
| `pages.jsonl` | JSONL | Full page content: body text, footnotes, page numbers |

### Pages JSONL Schema

```jsonc
{
  "page_id": 1,                    // Sequential page ID
  "book_id": 1,                    // FK to book
  "shamela_page_id": 1,           // Original Shamela ID
  "part": null,                    // Volume/part (for multi-volume books)
  "page_num": 3,                   // Printed page number
  "sequence_num": 1,              // Reading order
  "body": "Arabic text...",       // Main text with minimal HTML:
                                  // <span data-type='title' id=toc-N>Heading</span>
  "footnotes": null,              // Footnote text (¬١, ¬٢ markers)
  "hints": null,                  // Additional annotations
  "services_raw": null            // Source service data
}
```

### TOC JSONL Schema

```jsonc
{
  "title_id": 1,
  "book_id": 1,
  "page_id": 5,                   // Page where heading appears
  "parent_id": null,              // Hierarchical parent (null = root)
  "shamela_title_id": 2,
  "title_text": "مقدمة المحقق"    // Heading text
}
```

### Book Metadata Schema

```jsonc
{
  "book_id": 1,
  "shamela_id": 1,
  "title_ar": "الفواكه العذاب في الرد على من لم يحكم السنة والكتاب",
  "book_type": 1,
  "book_type_label": "كتاب",
  "category_id": 1,
  "category_name_ar": "العقيدة",
  "main_author_id": 513,
  "main_author_name_ar": "حمد بن ناصر آل معمر",
  "main_author_death_hijri": 1225,
  "authors": [
    { "author_id": 513, "role": "author", "name_ar": "...", "death_hijri": 1225 }
  ],
  "printed": true,
  "is_hidden": false,
  "betaka_text": "Full bibliographic description...",
  "version_major": 5,
  "version_minor": 0,
  "volume_count_observed": 0,
  "has_multi_part": false
}
```

## Statistics

| Metric | Count |
|--------|-------|
| **Books** | 8,589 |
| **Pages** | 7,611,186 |
| **TOC entries** | 4,060,485 |
| **Authors** | 3,187 |
| **Categories** | 41 |
| **Isnad (chains)** | 35,526 |
| **Narrators** | 18,989 |
| **Quran verses** | 6,236 |
| **Tafsir cross-refs** | 84,430 |
| **Hadith cross-refs** | 37,076 |
| **Root dictionary** | 1,952,803 |
| **Total size** | ~19 GB |
| **Meta directory** | ~177 MB |
| **Rejected books** | 0 |

### Largest Categories

| # | Category | Books |
|---|----------|-------|
| 06 | كتب السنة (Hadith Collections) | 1,241 |
| 01 | العقيدة (Creed) | 807 |
| 23 | الرقائق والآداب والأذكار (Spirituality) | 625 |
| 26 | التراجم والطبقات (Biographies) | 579 |
| 19 | مسائل فقهية (Fiqh Issues) | 429 |
| 32 | الأدب (Literature) | 406 |
| 39 | كتب عامة (General) | 357 |
| 10 | علوم الحديث (Hadith Sciences) | 320 |

## Coverage

All major Islamic disciplines:

- **Quranic:** Tafsir, Quran Sciences, Tajwid & Qira'at
- **Hadith:** The six canonical collections (Kutub al-Sittah), Musnad Ahmad, Muwatta Malik, plus commentaries, sciences (`ilal, rijāl, `ilal), takhrij
- **Fiqh:** All four Sunni madhahib (Hanafi, Maliki, Shafi'i, Hanbali), general fiqh, usul al-fiqh, fiqh maxims, fatāwā
- **Aqeedah:** Classical creedal works, refutations, theological debates
- **Arabic Language:** Grammar (nahw), morphology (sarf), rhetoric (balāgha), lexicons, poetry, prosody
- **History & Biography:** Sirah, tarikh, tabaqat, genealogy, geography
- **Other:** Logic, medicine, general works

## Data Notes

- **Text is raw** — minimal HTML markup preserved (`<span data-type='title'>`, `<hr>`, `<table>`)
- **Footnote markers** — `(¬N)` notation linking body to footnotes
- **Quranic text** — bracketed with `﴿...﴾` markers
- **Honorifics** — ﷺ (SAW), ﵁, ﷿, etc. are single Unicode codepoints
- **Multi-volume works** — linked via `parent_id` / `group_id` in metadata
- **Author death dates** — Hijri years; negative values = CE dates
- **Hidden books** — `is_hidden: true` indicates copyright/access restrictions
- **`\r` line endings** — Shamela uses `\r` (carriage return) instead of `\n`
- **Base64 images** — some pages contain inline `<img>` with base64 data

## Extraction

- **Source:** Production PostgreSQL 16 (dockerized)
- **Method:** psycopg server-side cursor stream, constant memory
- **Date:** 2026-04-26
- **Extractor:** shamela-extractor v0.3.0
- **Snapshot:** `shamela4_full_extraction_pg_2026_04_26`
- **No books rejected** — all 8,589 books extracted successfully

## Usage Examples

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

# Load a single book by filtering
ds = load_dataset("AuthenticIlm/Shamela4_Full_DB", split="full")
book_1 = ds.filter(lambda x: x["book_id"] == 1)

# Or stream individual JSONL files directly
import json
with open("01__العقيدة/1__الفواكه-العذاب/pages.jsonl") as f:
    for line in f:
        page = json.loads(line)
        print(page["body"][:200])
```

### Load metadata with Polars

```python
import polars as pl

df = pl.read_parquet("_meta/book_metadata.parquet")
df.filter(pl.col("category_id") == 3)  # All tafsir books
```

### Query roots from the root dictionary

```python
df = pl.read_parquet("_meta/root_dictionary.parquet")
df.filter(pl.col("token").str.contains("صلو"))
```

## Intended Uses

- **Classical Islamic text research** — full-text search, citation extraction, cross-book linking
- **Arabic NLP** — language modeling on classical Arabic (pre-modern, diacritized, diverse genres)
- **Hadith studies** — chain-of-transmission analysis, narrator networks, takhrij
- **Intertextuality analysis** — quotation detection, author attribution, topic propagation
- **Lexicography** — root extraction, semantic field analysis from medieval dictionaries
- **LLM training/evaluation** — high-quality classical Arabic data for fine-tuning or evaluation

## Limitations

- This is a full DB dump, including the encoded Narrator tables, etc.

## License

This dataset is derived from **al-Maktaba al-Shamela (الشاملة)**, a free digital library of classical Islamic texts. The original texts are in the public domain. The compilation and digitization rights belong to their respective owners.

**Terms of use:** Research and personal use. Respect the intellectual property rights of muhaqqiqīn (editors) and publishers for critical editions.

## Citation

```bibtex
@misc{shamela4_full_2026,
  title = {Shamela 4 — Full Islamic Library Corpus},
  author = {{AuthenticIlm}},
  year = {2026},
  note = {Extracted from al-Maktaba al-Shamela v4. 8,589 books, 7.6M pages.},
  url = {https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB}
}

@online{shamela,
  title = {al-Maktaba al-Shamela},
  url = {https://shamela.ws}
}
```

---
