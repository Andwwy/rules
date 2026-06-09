# Spec

## Annotation data source

The app reads the documents to annotate from a **`sample` table** and writes
annotations back to the same store. Both live in **MotherDuck** (cloud DuckDB),
selected via `MOTHERDUCK_TOKEN`. The token should be set **everywhere — local dev
and Render alike** — so all annotators read and write the same shared cloud DB. The
local-file fallback (`sample.db` / `annotations.db`, used when no token is present)
is an offline-only escape hatch, not the normal path.

Because the source documents live in MotherDuck, **no database is tracked in git**:
the Render deploy ships only code and connects to MotherDuck with the token.
`sample.db` stays gitignored (it's just the local seed for the uploader script).

### Current: fixed 100-file sample

For now the annotation set is a **frozen sample of 100 files**, stored in MotherDuck
as `rules.sample`. Everyone annotates the exact same fixed set.

Load / refresh it from the local `sample.db` seed:

```bash
python3 upload_sample_to_motherduck.py   # CREATE OR REPLACE rules.sample (100 files)
```

**`sample` table schema:**

| column | type | notes |
|---|---|---|
| `id` | VARCHAR | stable file id |
| `source_url` | VARCHAR | GitHub blob URL |
| `raw_url` | VARCHAR | raw content URL |
| `repo_name` | VARCHAR | `owner/repo` |
| `file_type` | VARCHAR | e.g. `yml`, `.clinerules`, `md` |
| `content` | VARCHAR | full file text (what gets annotated) |
| `content_len` | INTEGER | length of `content` |
| `source` | VARCHAR | how it was discovered (`awesome_list`, `github_search`, …) |

### Later: all scraped files

This 100-file sample is a stopgap. We will switch the annotation source to the
**full set of scraped files** once the dataset is ready. To switch, regenerate the
larger set (from the crawlers / `process.py`) and load it into the MotherDuck
`sample` table with the same schema — `annotate.py` reads `FROM sample`, so no code
change is needed.
