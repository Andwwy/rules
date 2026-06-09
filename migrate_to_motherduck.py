#!/usr/bin/env python3
"""One-off: copy all annotations from the local annotations.db into MotherDuck.

Run AFTER setting MOTHERDUCK_TOKEN (in .env or the shell). Idempotent — rows
with existing primary keys are skipped, so it's safe to run more than once.

  python3 migrate_to_motherduck.py
"""
import os, duckdb, importlib.util

# load .env so MOTHERDUCK_TOKEN can live there too
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _l in open(_env):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

token = os.environ.get("MOTHERDUCK_TOKEN") or os.environ.get("motherduck_token")
assert token, "Set MOTHERDUCK_TOKEN (in .env or the environment) first."
db = os.environ.get("MOTHERDUCK_DATABASE", "rules")

# reuse the app's schema definition
spec = importlib.util.spec_from_file_location("annotate", "annotate.py")
annotate = importlib.util.module_from_spec(spec); spec.loader.exec_module(annotate)

src = duckdb.connect("annotations.db", read_only=True)
dst = duckdb.connect(f"md:?motherduck_token={token}")
dst.execute(f"CREATE DATABASE IF NOT EXISTS {db}"); dst.execute(f"USE {db}")
annotate._ensure_schema(dst)

for t in ("extracted_rules", "annotators", "app_settings", "llm_runs", "saved_prompts"):
    try:
        cols = [d[0] for d in src.execute(f"SELECT * FROM {t} LIMIT 0").description]
        rows = src.execute(f"SELECT * FROM {t}").fetchall()
    except Exception as e:
        print(f"  {t}: skip ({e})"); continue
    if not rows:
        print(f"  {t}: 0 rows"); continue
    ph = ",".join(["?"] * len(cols))
    dst.executemany(f"INSERT OR IGNORE INTO {t} ({','.join(cols)}) VALUES ({ph})", rows)
    print(f"  {t}: copied {len(rows)} rows")

print("Done. Annotations are now in MotherDuck database:", db)
src.close(); dst.close()
