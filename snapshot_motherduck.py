#!/usr/bin/env python3
"""Snapshot / restore the MotherDuck `rules` database — run BEFORE a schema change.

  python3 snapshot_motherduck.py snapshot            # back up rules -> rules_v_<timestamp>
  python3 snapshot_motherduck.py list                # show snapshots
  python3 snapshot_motherduck.py restore rules_v_... # roll back rules from a snapshot
  python3 snapshot_motherduck.py drop rules_v_...    # delete a snapshot

A snapshot is a full independent copy (schema + data). Restoring overwrites the
live `rules` database, so STOP the app first (`docker compose stop`) before restore.
"""
import os, sys, duckdb

_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for l in open(_env):
        l = l.strip()
        if l and not l.startswith("#") and "=" in l:
            k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or os.environ.get("motherduck_token")
DB = os.environ.get("MOTHERDUCK_DATABASE", "rules")
assert TOKEN, "Set MOTHERDUCK_TOKEN (in .env) first."
con = duckdb.connect(f"md:?motherduck_token={TOKEN}")

cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"

if cmd == "snapshot":
    import datetime  # only place a timestamp is needed
    snap = f"{DB}_v_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    con.execute(f"DROP DATABASE IF EXISTS {snap}")
    con.execute(f"CREATE DATABASE {snap}")
    con.execute(f"COPY FROM DATABASE {DB} TO {snap}")
    n = con.execute(f"SELECT COUNT(*) FROM {snap}.main.extracted_rules").fetchone()[0]
    print(f"✓ snapshot created: {snap}  ({n} rules)")

elif cmd == "list":
    rows = [r[0] for r in con.execute("SHOW DATABASES").fetchall()
            if r[0].startswith(f"{DB}_v_")]
    print("snapshots:", rows or "(none)")

elif cmd == "restore" and len(sys.argv) > 2:
    snap = sys.argv[2]
    con.execute(f"USE {snap}")                 # so `rules` can be dropped
    con.execute(f"DROP DATABASE IF EXISTS {DB}")
    con.execute(f"CREATE DATABASE {DB}")
    con.execute(f"COPY FROM DATABASE {snap} TO {DB}")
    n = con.execute(f"SELECT COUNT(*) FROM {DB}.main.extracted_rules").fetchone()[0]
    print(f"✓ restored {DB} from {snap}  ({n} rules). Restart the app.")

elif cmd == "drop" and len(sys.argv) > 2:
    snap = sys.argv[2]
    con.execute(f"USE {DB}"); con.execute(f"DROP DATABASE IF EXISTS {snap}")
    print(f"✓ dropped snapshot {snap}")

else:
    print(__doc__)
con.close()
