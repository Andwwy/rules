#!/usr/bin/env python3
"""One-off: load the fixed 100-file sample (local sample.db) into MotherDuck.

The app reads source documents from the `sample` table. In production
(Render + MOTHERDUCK_TOKEN set) that table is served from MotherDuck, so the
local sample.db does NOT need to be tracked in git. Re-run to refresh the sample
(it replaces the table). We'll swap this fixed sample for the full scraped set
later — see SPEC.md.

  python3 upload_sample_to_motherduck.py
"""
import os, duckdb

_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _l in open(_env):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

token = os.environ.get("MOTHERDUCK_TOKEN") or os.environ.get("motherduck_token")
assert token, "Set MOTHERDUCK_TOKEN (in .env or the environment) first."
db = os.environ.get("MOTHERDUCK_DATABASE", "rules")
SAMPLE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.db")
assert os.path.exists(SAMPLE_DB), "local sample.db not found"

con = duckdb.connect(f"md:?motherduck_token={token}")
con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
con.execute(f"ATTACH '{SAMPLE_DB}' AS local_sample (READ_ONLY)")
con.execute(f"CREATE OR REPLACE TABLE {db}.sample AS SELECT * FROM local_sample.sample")
n = con.execute(f"SELECT count(*) FROM {db}.sample").fetchone()[0]
print(f"Loaded {n} files into MotherDuck {db}.sample")
con.close()
