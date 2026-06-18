#!/usr/bin/env python3
"""
annotate.py – Interactive rule annotation webapp.

Browse files from sample.db, manually select text to extract rules,
run Perplexity/OpenAI/Claude LLM prompts to auto-extract, and compare
extractions side-by-side with per-extractor colour coding.

Keyboard shortcuts:
  Cmd+Enter   Add selected text as rule
  j / k       Navigate rule cards
  n           Open note editor on focused rule
  d           Delete focused rule
  Enter       Save note
  Escape      Cancel note / clear selection

Usage:
  export PERPLEXITY_API_KEY=pplx-...
  python3 annotate.py
  open http://localhost:5002
"""

import hashlib, json, os, re, threading, time, urllib.request, urllib.error
import concurrent.futures, contextlib
from datetime import datetime, timezone
import duckdb
from flask import Flask, jsonify, request, make_response

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DB      = os.path.join(BASE_DIR, "sample.db")
ANNOT_DB       = os.path.join(BASE_DIR, "annotations.db")
PERPLEXITY_CHAT_URL  = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_AGENT_URL = "https://api.perplexity.ai/v1/agent"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    for _l in open(_env_path):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# DB
#
# Annotations live in MotherDuck (cloud DuckDB) when MOTHERDUCK_TOKEN is set —
# this is what makes the Render deployment (ephemeral filesystem) and multi-device
# work. With no token, it falls back to the local annotations.db file. The token is
# read from the environment (.env locally, Render env vars in production).
# ---------------------------------------------------------------------------
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or os.environ.get("motherduck_token")
# Canonical DB is the v2 set ("rules v2.0"). Override with MOTHERDUCK_DATABASE to point
# elsewhere (the old "rules" DB and snapshots remain attached but are no longer the default).
MOTHERDUCK_DATABASE = os.environ.get("MOTHERDUCK_DATABASE", "rules_v2_6_18")
USE_MOTHERDUCK = bool(MOTHERDUCK_TOKEN)

def _ensure_schema(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS extracted_rules (
            id           VARCHAR PRIMARY KEY,
            file_id      VARCHAR NOT NULL,
            rule_text    TEXT    NOT NULL,
            char_start   INTEGER,
            char_end     INTEGER,
            line_start   INTEGER,
            line_end     INTEGER,
            source       VARCHAR NOT NULL,
            llm_run_id   VARCHAR,
            notes        VARCHAR,
            extracted_by VARCHAR,
            created_at   TIMESTAMP DEFAULT now()
        )
    """)
    # notes == user "Comment" in the inspector;
    # annotator == which person owns this labeling (hand rules AND the LLM runs they triggered)
    # kind == 'rule' (a directive — gets a deontic tag) or 'context' (background, no tag)
    # context_type == sub-type of a 'context' node: 'condition' | 'reference' |
    # 'definition' (NULL for rules and for unclassified context).
    # accepted   == passed an Extract multi-judge vote (solid highlight) — always
    #               true for hand rules; candidate (sub-threshold) spans are NOT rows.
    # vote_count == # distinct judges that backed this item (ring numerator)
    # judge_total== # judges in the run that produced it (ring denominator)
    # tag        == the EFFECTIVE deontic tag (a human edit overrides the LLM's pick).
    # llm_tag     == the tag the Label vote decided, preserved so a human change can be
    #               contrasted/reverted. "Revised" = llm_tag set AND tag != llm_tag.
    for col in ("notes VARCHAR", "extracted_by VARCHAR",
                "tag VARCHAR", "power_type VARCHAR", "llm_rationale TEXT",
                "annotator VARCHAR", "kind VARCHAR", "context_type VARCHAR",
                "reviewed BOOLEAN",   # set when a human edits an LLM-extracted item's tag/kind/context
                "accepted BOOLEAN", "vote_count INTEGER", "judge_total INTEGER",
                "llm_tag VARCHAR"):
        try: con.execute(f"ALTER TABLE extracted_rules ADD COLUMN {col}")
        except Exception: pass
    con.execute("CREATE TABLE IF NOT EXISTS annotators (name VARCHAR PRIMARY KEY, created_at TIMESTAMP DEFAULT now())")
    con.execute("CREATE TABLE IF NOT EXISTS app_settings (key VARCHAR PRIMARY KEY, value TEXT)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_runs (
            id VARCHAR PRIMARY KEY, file_id VARCHAR NOT NULL, prompt TEXT, model VARCHAR,
            raw_response TEXT, rule_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT now()
        )
    """)
    con.execute("CREATE TABLE IF NOT EXISTS saved_prompts (id VARCHAR PRIMARY KEY, prompt TEXT NOT NULL, label VARCHAR, saved_at TIMESTAMP DEFAULT now())")
    # One free-text comment per (file, annotator) — the file-level "Comment" panel.
    con.execute("""
        CREATE TABLE IF NOT EXISTS file_comments (
            file_id    VARCHAR NOT NULL,
            annotator  VARCHAR NOT NULL,
            comment    TEXT,
            updated_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (file_id, annotator)
        )
    """)
    # A relation is a DIRECTED, one-to-one edge between two entities in
    # extracted_rules (each a 'rule' or 'context' — relations are NOT a `kind`,
    # they connect kinds). source_id → target_id is one-way; the PK dedupes an
    # identical edge per annotator. Per-annotator isolation mirrors extracted_rules.
    con.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id            VARCHAR PRIMARY KEY,
            file_id       VARCHAR NOT NULL,
            source_id     VARCHAR NOT NULL,
            target_id     VARCHAR NOT NULL,
            relation_type VARCHAR,
            notes         VARCHAR,
            source        VARCHAR DEFAULT 'hand',
            llm_run_id    VARCHAR,
            llm_rationale TEXT,
            annotator     VARCHAR,
            created_at    TIMESTAMP DEFAULT now()
        )
    """)
    # relation_type holds the USER's edge label; llm_relation_type preserves the
    # label an LLM originally suggested for the same edge, so the two can be
    # contrasted. Effective/displayed label = relation_type (user) ?? llm_relation_type.
    try: con.execute("ALTER TABLE relations ADD COLUMN llm_relation_type VARCHAR")
    except Exception: pass
    # accepted/vote_count/judge_total mirror extracted_rules — set when a relation
    # passes the multi-judge Relation vote (hand relations: accepted, null counts).
    for col in ("accepted BOOLEAN", "vote_count INTEGER", "judge_total INTEGER"):
        try: con.execute(f"ALTER TABLE relations ADD COLUMN {col}")
        except Exception: pass
    # ── Multi-judge voting ──────────────────────────────────────────────────
    # judge_runs: one row per Extract/Label/Relation multi-judge pass.
    # judge_votes: every individual judge output (powers the vote rings, the
    #   inspection panel, and the Extract candidate/light highlights).
    con.execute("""
        CREATE TABLE IF NOT EXISTS judge_runs (
            id VARCHAR PRIMARY KEY, file_id VARCHAR NOT NULL, annotator VARCHAR,
            mode VARCHAR NOT NULL, threshold INTEGER, judge_total INTEGER,
            models TEXT, created_at TIMESTAMP DEFAULT now()
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS judge_votes (
            id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, file_id VARCHAR NOT NULL,
            annotator VARCHAR, mode VARCHAR NOT NULL,
            judge_idx INTEGER, model VARCHAR,
            char_start INTEGER, char_end INTEGER, kind VARCHAR,
            rule_id VARCHAR, source_id VARCHAR, target_id VARCHAR,
            value VARCHAR, rationale TEXT, created_at TIMESTAMP DEFAULT now()
        )
    """)

def _md_connect(target):
    # On Render, point DuckDB's extension/home dir at the writable /tmp so the
    # MotherDuck extension can auto-install regardless of the home-dir setup.
    cfg = {"home_directory": "/tmp"} if os.environ.get("RENDER") else {}
    return duckdb.connect(target, config=cfg)

_annot_base = None          # cached MotherDuck session (one network connection)
_md_lock = threading.Lock()  # guards the lazy session init against concurrent requests

def _md_base():
    """Open once and cache the single MotherDuck session. Both the annotation
    tables AND the `sample` source documents live in this one cloud database, so
    nothing DB-related needs to ship in git for the Render deploy.

    The init is lock-guarded: a fresh page load fires several requests at once
    (threaded dev server / gunicorn --threads), and without the lock they'd race
    on session creation and some would come back empty."""
    global _annot_base
    if _annot_base is None:
        with _md_lock:
            if _annot_base is None:   # double-checked: another thread may have won
                boot = _md_connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")
                boot.execute(f"CREATE DATABASE IF NOT EXISTS {MOTHERDUCK_DATABASE}")
                boot.close()
                base = _md_connect(f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}")
                _ensure_schema(base)
                _annot_base = base
    return _annot_base

def annot_con():
    """Return an annotations connection. Route code uses it then calls .close().
    - Local: a fresh per-request connection to annotations.db (cheap, thread-safe).
    - MotherDuck: a per-request cursor over one cached cloud session; .close()
      then closes only the cursor, keeping the (expensive) session alive."""
    if USE_MOTHERDUCK:
        return _md_base().cursor()
    con = duckdb.connect(ANNOT_DB)
    _ensure_schema(con)
    return con

_sample_ro = None
def sample_con():
    """Read-only handle to the `sample` source documents — callers need NOT close it.
    - MotherDuck: a FRESH cursor over the shared cloud session per call. A cursor is
      duckdb's thread-safe unit, so each request gets its own (a single shared cursor
      would race across threads and return empty). The `sample` table lives in the
      same MotherDuck database; load it with upload_sample_to_motherduck.py.
    - Local: a cached read-only connection to sample.db."""
    global _sample_ro
    if USE_MOTHERDUCK:
        return _md_base().cursor()
    if _sample_ro is None and os.path.exists(SAMPLE_DB):
        _sample_ro = duckdb.connect(SAMPLE_DB, read_only=True)
    return _sample_ro

def make_id(*parts):
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Routes – files
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    # The whole app is this one inline HTML/JS/CSS page, and it changes often.
    # Tell the browser never to reuse a cached copy, so a refresh always loads the
    # current code (stale cached pages were showing old, broken UI states).
    resp = make_response(HTML)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/api/files")
def api_files():
    sc = sample_con()
    if not sc:
        return jsonify({"error": "sample.db not found – run extract_sample.py first"}), 404
    rows = sc.execute(
        "SELECT id, source_url, repo_name, file_type, content_len, source "
        "FROM sample ORDER BY content_len DESC"
    ).fetchall()
    annotator = (request.args.get("annotator") or "").strip()
    con = annot_con()
    if annotator:
        raw = con.execute(
            "SELECT file_id, source, COUNT(*) FROM extracted_rules WHERE annotator=? GROUP BY 1,2",
            [annotator]
        ).fetchall()
    else:
        raw = con.execute(
            "SELECT file_id, source, COUNT(*) FROM extracted_rules GROUP BY 1,2"
        ).fetchall()
    con.close()
    counts = {}
    for fid, src, n in raw:
        counts.setdefault(fid, {})[src] = n
    # "machine" = everything the judge produced — extract ('llm') AND revise — so the
    # count badge survives a revise pass (which swaps 'llm' rows for 'revise' rows).
    def _machine(d):
        return sum(n for s, n in d.items() if s != "hand")
    return jsonify([{
        "id": r[0], "source_url": r[1], "repo_name": r[2],
        "file_type": r[3], "content_len": r[4], "source": r[5],
        "hand_count": counts.get(r[0], {}).get("hand", 0),
        "llm_count":  _machine(counts.get(r[0], {})),
        # revised == a 'revise' pass has been run (badge turns green vs. blue for extract-only)
        "revised":    "revise" in counts.get(r[0], {}),
    } for r in rows])

@app.route("/api/file/<fid>")
def api_file(fid):
    sc = sample_con()
    if not sc: return jsonify({"error": "sample.db not found"}), 404
    row = sc.execute(
        "SELECT id, source_url, raw_url, repo_name, file_type, content, content_len, source "
        "FROM sample WHERE id=?", [fid]
    ).fetchone()
    if not row: return jsonify({"error": "not found"}), 404
    return jsonify({
        "id": row[0], "source_url": row[1], "raw_url": row[2],
        "repo_name": row[3], "file_type": row[4],
        "content": row[5] or "", "content_len": row[6], "source": row[7],
    })

# ---------------------------------------------------------------------------
# Routes – file-level comments (one per file per annotator)
# ---------------------------------------------------------------------------
@app.route("/api/file-comment/<fid>")
def get_file_comment(fid):
    annotator = (request.args.get("annotator") or "").strip()
    con = annot_con()
    row = con.execute(
        "SELECT comment FROM file_comments WHERE file_id=? AND annotator=?",
        [fid, annotator]
    ).fetchone()
    con.close()
    return jsonify({"comment": (row[0] if row else "") or ""})

@app.route("/api/file-comment/<fid>", methods=["POST"])
def save_file_comment(fid):
    b = request.json or {}
    annotator = (b.get("annotator") or "").strip()
    comment   = (b.get("comment") or "").strip()
    if not annotator:
        return jsonify({"error": "annotator required"}), 400
    con = annot_con()
    if comment:
        con.execute("""
            INSERT INTO file_comments(file_id, annotator, comment, updated_at)
            VALUES(?,?,?,now())
            ON CONFLICT(file_id, annotator)
            DO UPDATE SET comment=excluded.comment, updated_at=now()
        """, [fid, annotator, comment])
    else:
        # an emptied comment removes the row (keeps the table clean)
        con.execute("DELETE FROM file_comments WHERE file_id=? AND annotator=?", [fid, annotator])
    con.close()
    return jsonify({"ok": True, "comment": comment})

# ---------------------------------------------------------------------------
# Routes – rules
# ---------------------------------------------------------------------------
def _rule_row(r):
    return {
        "id": r[0], "rule_text": r[1],
        "char_start": r[2], "char_end": r[3],
        "line_start": r[4], "line_end": r[5],
        "source": r[6], "llm_run_id": r[7],
        "notes": r[8], "extracted_by": r[9],
        "created_at": str(r[10]),
        "tag": r[11], "power_type": r[12], "llm_rationale": r[13],
        "annotator": r[14], "kind": r[15] or "rule",
        "context_type": r[16], "reviewed": bool(r[17]),
        # accepted: NULL (legacy / hand) is treated as accepted; only explicit
        # False marks a sub-threshold candidate (currently candidates aren't rows).
        "accepted": (r[18] is not False),
        "vote_count": r[19], "judge_total": r[20],
        "llm_tag": r[21],   # the Label vote's pick, preserved under a human override
    }

@app.route("/api/all-rules")
def api_all_rules():
    """All extracted rules joined with sample metadata for CSV export."""
    acon = annot_con()
    rows = acon.execute("""
        SELECT r.id, r.file_id, r.rule_text, r.line_start, r.line_end,
               r.source, r.extracted_by, r.notes, r.created_at
        FROM extracted_rules r
        ORDER BY r.file_id, COALESCE(r.line_start, 999999), r.created_at
    """).fetchall()
    acon.close()
    # join source_url from sample.db
    sc = sample_con()
    url_map = {}
    if rows:
        fids = list({r[1] for r in rows})
        placeholders = ",".join("?" * len(fids))
        url_map = {r[0]: r[1] for r in sc.execute(
            f"SELECT id, source_url FROM sample WHERE id IN ({placeholders})", fids
        ).fetchall()}
    # NOTE: sample_con() is a cached, shared read-only connection — do NOT close it.
    return jsonify([{
        "id": r[0], "file_id": r[1],
        "source_url": url_map.get(r[1], ""),
        "rule_text": r[2], "line_start": r[3], "line_end": r[4],
        "source": r[5], "extracted_by": r[6], "notes": r[7], "created_at": str(r[8]),
    } for r in rows])

@app.route("/api/rules/<fid>")
def api_rules(fid):
    annotator = (request.args.get("annotator") or "").strip()
    con = annot_con()
    sql = """
        SELECT id, rule_text, char_start, char_end, line_start, line_end,
               source, llm_run_id, notes, extracted_by, created_at,
               tag, power_type, llm_rationale, annotator, kind, context_type,
               reviewed, accepted, vote_count, judge_total, llm_tag
        FROM extracted_rules WHERE file_id=?
    """
    params = [fid]
    if annotator:
        sql += " AND annotator=?"
        params.append(annotator)
    sql += " ORDER BY COALESCE(line_start, 999999), COALESCE(char_start, 999999), created_at"
    rows = con.execute(sql, params).fetchall()
    con.close()
    return jsonify([_rule_row(r) for r in rows])

@app.route("/api/rules", methods=["POST"])
def save_rule():
    b = request.json or {}
    fid       = b.get("file_id","").strip()
    rule_text = b.get("rule_text","").strip()
    if not fid or not rule_text:
        return jsonify({"error": "file_id and rule_text required"}), 400
    annotator = (b.get("annotator") or b.get("extracted_by") or "unknown").strip()
    # Include the selection's char position so the SAME text selected at a DIFFERENT
    # spot is a DISTINCT rule (duplicates are independent). Same text + same position
    # still dedupes (it's literally the same selection).
    rid = make_id(fid, "hand", annotator, rule_text, b.get("char_start"), b.get("char_end"))
    by  = b.get("extracted_by") or annotator
    kind = "context" if (b.get("kind") == "context") else "rule"
    con = annot_con()
    con.execute("""
        INSERT OR IGNORE INTO extracted_rules
            (id,file_id,rule_text,char_start,char_end,line_start,line_end,
             source,llm_run_id,notes,extracted_by,annotator,kind,accepted)
        VALUES(?,?,?,?,?,?,?,'hand',NULL,NULL,?,?,?,TRUE)
    """, [rid, fid, rule_text,
          b.get("char_start"), b.get("char_end"),
          b.get("line_start"), b.get("line_end"), by, annotator, kind])
    if annotator:
        con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
    con.close()
    return jsonify({
        "id": rid, "rule_text": rule_text,
        "char_start": b.get("char_start"), "char_end": b.get("char_end"),
        "line_start": b.get("line_start"), "line_end": b.get("line_end"),
        "source": "hand", "llm_run_id": None, "notes": None,
        "extracted_by": by, "annotator": annotator, "kind": kind,
        "tag": None, "power_type": None, "llm_rationale": None,
        "context_type": None, "reviewed": False,
        "accepted": True, "vote_count": None, "judge_total": None, "llm_tag": None,
    })

@app.route("/api/rules/<rid>", methods=["PATCH"])
def patch_rule(rid):
    """Update notes/tag/power_type/llm_rationale/kind/context_type/reviewed, or the
    rule's text + position (rule_text, char_start/end, line_start/end) when re-selected."""
    b = request.json or {}
    updates, params = [], []
    for k in ("notes", "tag", "power_type", "llm_rationale", "kind", "context_type", "reviewed", "rule_text"):
        if k in b:
            updates.append(f"{k}=?")
            params.append(b.get(k) or None)
    for k in ("char_start", "char_end", "line_start", "line_end"):   # numeric: keep 0, don't coerce
        if k in b:
            updates.append(f"{k}=?")
            params.append(b.get(k))
    if not updates:
        return jsonify({"ok": True})
    params.append(rid)
    con = annot_con()
    con.execute(f"UPDATE extracted_rules SET {', '.join(updates)} WHERE id=?", params)
    con.close()
    return jsonify({"ok": True})

@app.route("/api/rules/<rid>", methods=["DELETE"])
def delete_rule(rid):
    con = annot_con()
    con.execute("DELETE FROM extracted_rules WHERE id=?", [rid])
    # An entity can't be half of an edge once it's gone — drop relations touching it.
    con.execute("DELETE FROM relations WHERE source_id=? OR target_id=?", [rid, rid])
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Routes – relations (directed one-to-one edges between two entities)
# ---------------------------------------------------------------------------
def _relation_row(r):
    return {
        "id": r[0], "file_id": r[1], "source_id": r[2], "target_id": r[3],
        "relation_type": r[4], "notes": r[5], "source": r[6],
        "llm_run_id": r[7], "llm_rationale": r[8], "annotator": r[9],
        "created_at": str(r[10]), "llm_relation_type": r[11],
        "accepted": (r[12] is not False), "vote_count": r[13], "judge_total": r[14],
    }

_REL_COLS = ("id, file_id, source_id, target_id, relation_type, notes, source, "
             "llm_run_id, llm_rationale, annotator, created_at, llm_relation_type, "
             "accepted, vote_count, judge_total")

@app.route("/api/relations/<fid>")
def api_relations(fid):
    annotator = (request.args.get("annotator") or "").strip()
    con = annot_con()
    sql = f"SELECT {_REL_COLS} FROM relations WHERE file_id=?"
    params = [fid]
    if annotator:
        sql += " AND annotator=?"
        params.append(annotator)
    sql += " ORDER BY created_at"
    rows = con.execute(sql, params).fetchall()
    con.close()
    return jsonify([_relation_row(r) for r in rows])

@app.route("/api/relations", methods=["POST"])
def save_relation():
    b = request.json or {}
    fid = (b.get("file_id") or "").strip()
    src = (b.get("source_id") or "").strip()
    tgt = (b.get("target_id") or "").strip()
    if not fid or not src or not tgt:
        return jsonify({"error": "file_id, source_id and target_id required"}), 400
    if src == tgt:
        return jsonify({"error": "a relation can't connect an entity to itself"}), 400
    annotator = (b.get("annotator") or "unknown").strip()
    con = annot_con()
    # Both endpoints must be real entities in this file (and this annotator's set).
    have = con.execute(
        "SELECT id FROM extracted_rules WHERE file_id=? AND id IN (?,?)"
        " AND annotator IS NOT DISTINCT FROM ?",
        [fid, src, tgt, annotator or None],
    ).fetchall()
    if {r[0] for r in have} != {src, tgt}:
        con.close()
        return jsonify({"error": "source_id and target_id must both be entities in this file"}), 400
    rtype = (b.get("relation_type") or None)       # user's label
    llm_rtype = (b.get("llm_relation_type") or None)  # an LLM's suggested label (if any)
    edge_source = "llm" if (b.get("source") == "llm") else "hand"
    # directed + per-annotator dedupe, keyed by the CREATION-TIME type so the same
    # pair can carry more than one typed relation (e.g. "refinement" and "conflict").
    # Note: a later PATCH of relation_type does NOT change this id (refs stay stable),
    # so two rows could in theory converge to the same effective type — tolerated,
    # since multiple typed edges per pair are intentionally allowed.
    rid = make_id(fid, src, tgt, (rtype or llm_rtype or ""), annotator)
    con.execute(
        "INSERT OR IGNORE INTO relations"
        " (id,file_id,source_id,target_id,relation_type,notes,source,llm_run_id,"
        "  llm_rationale,annotator,llm_relation_type)"
        " VALUES (?,?,?,?,?,?,?,NULL,?,?,?)",
        [rid, fid, src, tgt, rtype, b.get("notes") or None, edge_source,
         b.get("llm_rationale") or None, annotator or None, llm_rtype],
    )
    if annotator:
        con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
    row = con.execute(f"SELECT {_REL_COLS} FROM relations WHERE id=?", [rid]).fetchone()
    con.close()
    return jsonify(_relation_row(row))

@app.route("/api/relations/<rid>", methods=["PATCH"])
def patch_relation(rid):
    """Update relation_type / notes / endpoints (source_id, target_id) on an edge."""
    b = request.json or {}
    updates, params = [], []
    for k in ("relation_type", "notes", "source_id", "target_id"):
        if k in b:
            updates.append(f"{k}=?")
            params.append(b.get(k) or None)
    if not updates:
        return jsonify({"ok": True})
    params.append(rid)
    con = annot_con()
    con.execute(f"UPDATE relations SET {', '.join(updates)} WHERE id=?", params)
    con.close()
    return jsonify({"ok": True})

@app.route("/api/relations/<rid>", methods=["DELETE"])
def delete_relation(rid):
    con = annot_con()
    con.execute("DELETE FROM relations WHERE id=?", [rid])
    con.close()
    return jsonify({"ok": True})

@app.route("/api/relations", methods=["DELETE"])
def delete_relations_bulk():
    """Delete every relation in a file for one annotator (the 'Delete all' button)."""
    fid = (request.args.get("file_id") or "").strip()
    annotator = (request.args.get("annotator") or "").strip()
    if not fid:
        return jsonify({"error": "file_id required"}), 400
    con = annot_con()
    con.execute(
        "DELETE FROM relations WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?",
        [fid, annotator or None])
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Routes – LLM
# ---------------------------------------------------------------------------
def _perplexity_complete(model, sys_msg, usr_msg, api_key, max_out=16000):
    """Single completion call. Third-party models (anthropic/*, openai/*) use the
    Agent API; native Sonar models use the Chat API. Returns the raw text."""
    use_agent = "/" in model
    if use_agent:
        payload = {"model": model, "input": usr_msg, "max_output_tokens": max_out}
        if sys_msg:
            payload["instructions"] = sys_msg
        data = json.dumps(payload).encode(); url = PERPLEXITY_AGENT_URL
    else:
        messages = [{"role": "user", "content": usr_msg}]
        if sys_msg:
            messages.insert(0, {"role": "system", "content": sys_msg})
        data = json.dumps({"model": model, "messages": messages,
                           "temperature": 0.1, "max_tokens": max_out}).encode()
        url = PERPLEXITY_CHAT_URL
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        rd = json.loads(resp.read())
    if use_agent:
        try: return rd["output"][0]["content"][0]["text"]
        except (KeyError, IndexError): return ""
    return rd.get("choices", [{}])[0].get("message", {}).get("content", "")

_REL_TYPES = ("refinement", "exception", "define", "checkpoint", "conflict", "duplication", "trigger")

@app.route("/api/llm-relations", methods=["POST"])
def run_llm_relations():
    """LLM judge in RELATION mode: propose typed edges between this annotator's
    existing rule/context entities. Stored with source='llm' and the LLM's type in
    llm_relation_type (relation_type/user label stays NULL) so the user can override
    and the LLM-vs-user contrast lights up."""
    b         = request.json or {}
    fid       = (b.get("file_id") or "").strip()
    prompt    = (b.get("prompt") or "").strip()
    model     = (b.get("model") or "gpt-4o-mini").strip()
    annotator = (b.get("annotator") or "").strip()
    if not fid or not prompt:
        return jsonify({"error": "file_id and prompt required"}), 400
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        return jsonify({"error": "PERPLEXITY_API_KEY not set"}), 400
    sc = sample_con()
    row = sc.execute("SELECT content FROM sample WHERE id=?", [fid]).fetchone()
    if not row:
        return jsonify({"error": "file not found"}), 404
    content = row[0] or ""

    con = annot_con()
    nodes = con.execute(
        "SELECT id, kind, tag, rule_text, line_start FROM extracted_rules"
        " WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?"
        " ORDER BY COALESCE(line_start, 999999), COALESCE(char_start, 999999)",
        [fid, annotator or None],
    ).fetchall()
    if len(nodes) < 2:
        con.close()
        return jsonify({"error": "Need at least 2 rules/context in this file to relate."}), 400

    # Give each entity a short stable label (R1, R2, …) the LLM can reference.
    label_to_id, lines = {}, []
    for i, (nid, kind, tag, text, ls) in enumerate(nodes, 1):
        lab = f"R{i}"; label_to_id[lab] = nid
        kindlab = (kind or "rule") + (f"/{tag}" if tag else "")
        lines.append(f'{lab} [{kindlab}] (line {ls}): {(text or "")[:200].strip()}')
    node_block = "\n".join(lines)

    sys_msg = ("You propose directed relations between the listed entities and return "
               "ONLY a single valid JSON array — no prose, no markdown fences. Each "
               "element uses the exact field schema in the user's instructions.")
    usr_msg = (f"{prompt}\n\nENTITIES in this document (reference them by their R-label):\n"
               f"{node_block}\n\nDocument:\n---\n{content[:25000]}\n---\n\n"
               "Return ONLY a JSON array of relations.")
    try:
        raw = _perplexity_complete(model, sys_msg, usr_msg, api_key, max_out=16000)
    except urllib.error.HTTPError as e:
        con.close(); return jsonify({"error": f"API {e.code}: {e.read().decode()}"}), 502
    except Exception as e:
        con.close(); return jsonify({"error": str(e)}), 502

    edges = _parse_llm_rules(raw)   # generic JSON-array parser (robust to truncation)
    run_id = make_id(fid, prompt, "rel", datetime.now(timezone.utc).isoformat())
    if annotator:
        con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
    # A relation judge run replaces this annotator's previous LLM relations for the
    # file (hand relations are kept).
    if b.get("replace_llm"):
        con.execute(
            "DELETE FROM relations WHERE file_id=? AND source='llm'"
            " AND annotator IS NOT DISTINCT FROM ?",
            [fid, annotator or None])

    saved, seen = [], set()
    for e in edges:
        s_lab = str(e.get("source") or e.get("source_id") or "").strip()
        t_lab = str(e.get("target") or e.get("target_id") or "").strip()
        src, tgt = label_to_id.get(s_lab), label_to_id.get(t_lab)
        if not src or not tgt or src == tgt:
            continue
        rtype = (e.get("type") or e.get("relation_type") or "").strip().lower()
        if rtype not in _REL_TYPES:
            # legacy "support (defines/…)" and any "define …" variants → 'define'
            rtype = "define" if rtype.startswith(("support", "define")) else None
        rationale = e.get("rationale") or e.get("llm_rationale")
        rid = make_id(fid, src, tgt, rtype or "", annotator, "llm")
        if rid in seen:
            continue
        seen.add(rid)
        con.execute(
            "INSERT OR IGNORE INTO relations"
            " (id,file_id,source_id,target_id,relation_type,notes,source,llm_run_id,"
            "  llm_rationale,annotator,llm_relation_type)"
            " VALUES (?,?,?,?,NULL,NULL,'llm',?,?,?,?)",
            [rid, fid, src, tgt, run_id, rationale, annotator or None, rtype])
        saved.append({"id": rid, "source_id": src, "target_id": tgt,
                      "relation_type": None, "llm_relation_type": rtype})
    con.close()
    return jsonify({"run_id": run_id, "count": len(saved), "relations": saved, "raw_response": raw})

@app.route("/api/llm", methods=["POST"])
def run_llm():
    b       = request.json or {}
    fid     = b.get("file_id","").strip()
    prompt  = b.get("prompt","").strip()
    model   = b.get("model","gpt-4o-mini").strip()
    annotator = (b.get("annotator") or "").strip()
    # Which judge pass this is — both extract rules AND context in one go, each
    # item self-labels its `kind`. The pass only differs by provenance/colour:
    #   'llm'    — fresh extraction (blue highlights)
    #   'revise' — refine using THIS annotator's human labels (still blue); needs labels
    pass_source = (b.get("source") or "llm").strip().lower()
    if pass_source not in ("llm", "revise"):
        pass_source = "llm"
    if not fid or not prompt:
        return jsonify({"error": "file_id and prompt required"}), 400
    api_key = os.environ.get("PERPLEXITY_API_KEY","")
    if not api_key:
        return jsonify({"error": "PERPLEXITY_API_KEY not set"}), 400
    sc = sample_con()
    row = sc.execute("SELECT content FROM sample WHERE id=?", [fid]).fetchone()
    if not row: return jsonify({"error": "file not found"}), 404
    content = row[0] or ""

    is_template = "{rule text}" in prompt
    if is_template:
        rule_id = b.get("rule_id", "").strip()
        if not rule_id:
            return jsonify({"error": "Focus a rule before running classification"}), 400
        acon = annot_con()
        rrow = acon.execute(
            "SELECT rule_text, line_start, line_end FROM extracted_rules WHERE id=?", [rule_id]
        ).fetchone()
        acon.close()
        if not rrow:
            return jsonify({"error": "Rule not found"}), 404
        rule_text_val, ls, le = rrow
        lines = content.split("\n")
        ctx_start = max(0, (ls or 1) - 1 - 20)
        ctx_end   = min(len(lines), (le or ls or 1) + 20)
        context_val = "\n".join(lines[ctx_start:ctx_end])
        import re as _re
        filled = prompt.replace("{rule text}", rule_text_val)
        filled = _re.sub(r"\{insert surrounding[^}]*\}", context_val, filled)
        sys_msg = ""
        usr_msg = filled
    else:
        sys_msg = (
            "You extract rules and context spans from the given document and return the "
            "result as a single valid JSON array and nothing else — no prose, no markdown "
            "fences, no trailing commentary. Use exactly the field schema given in the "
            "user's instructions for each array element."
        )
        # Revise feeds the annotator's human labels in as ground truth to refine around.
        human_section = ""
        if pass_source == "revise":
            hcon = annot_con()
            hrows = hcon.execute(
                "SELECT rule_text, kind, tag, line_start, line_end FROM extracted_rules"
                " WHERE file_id=? AND source='hand' AND annotator IS NOT DISTINCT FROM ?"
                " ORDER BY COALESCE(line_start, 999999)",
                [fid, annotator or None],
            ).fetchall()
            hcon.close()
            if not hrows:
                return jsonify({"error": "Revise needs human labels in this file first."}), 400
            labels = [{"rule_text": r[0], "kind": r[1] or "rule", "tag": r[2],
                       "line_start": r[3], "line_end": r[4]} for r in hrows]
            human_section = (
                "\n\nHUMAN LABELS for this document — primary guidance and the source of truth "
                "for what counts as a rule/context and at what granularity. STRONGLY prefer to "
                "keep every one: by default reproduce it with the same quote, kind, and tag, "
                "even if it looks vague, trivial, or incomplete (a short directive like "
                "\"Attempt a real fix\" is still a rule if the human marked it). You MAY leave "
                "out or adjust a human label only when you are confident it is genuinely wrong "
                "or nonsensical — that should be rare, and note why in its rationale. Use these "
                "labels as the bar for finding comparable items the human missed, and add "
                "those too.\n" + json.dumps(labels))
        usr_msg = (
            f"{prompt}{human_section}\n\nFile content:\n---\n{content[:30000]}\n---\n\nReturn ONLY a JSON array."
        )

    # Third-party models (anthropic/*, openai/*) use the Agent API; native Sonar models use Chat API
    use_agent_api = "/" in model
    if use_agent_api:
        # Agent API uses max_output_tokens; raise it so large rule sets aren't
        # truncated mid-JSON (truncation breaks the array parse).
        payload_dict = {"model": model, "input": usr_msg, "max_output_tokens": 32000}
        if sys_msg:
            payload_dict["instructions"] = sys_msg
        payload = json.dumps(payload_dict).encode()
        url = PERPLEXITY_AGENT_URL
    else:
        messages = [{"role": "user", "content": usr_msg}]
        if sys_msg:
            messages.insert(0, {"role": "system", "content": sys_msg})
        payload = json.dumps({
            "model": model, "messages": messages,
            "temperature": 0.1, "max_tokens": 16000,
        }).encode()
        url = PERPLEXITY_CHAT_URL

    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp_data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return jsonify({"error": f"API {e.code}: {e.read().decode()}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if use_agent_api:
        try:
            raw = resp_data["output"][0]["content"][0]["text"]
        except (KeyError, IndexError):
            raw = ""
    else:
        raw = resp_data.get("choices",[{}])[0].get("message",{}).get("content","")

    run_id = make_id(fid, prompt, datetime.now(timezone.utc).isoformat())
    con = annot_con()

    if is_template:
        # Classification mode: store raw JSON, attach rationale to the rule
        con.execute(
            "INSERT INTO llm_runs(id,file_id,prompt,model,raw_response,rule_count) VALUES(?,?,?,?,?,?)",
            [run_id, fid, prompt, model, raw, 0]
        )
        rule_id = b.get("rule_id", "").strip()
        if rule_id:
            con.execute(
                "UPDATE extracted_rules SET llm_rationale=? WHERE id=?", [raw, rule_id]
            )
        con.close()
        # Parse classification JSON for the response
        import re as _re2
        try:
            classification = json.loads(raw)
        except Exception:
            m = _re2.search(r"\{.*\}", raw, _re2.DOTALL)
            try:
                classification = json.loads(m.group()) if m else {}
            except Exception:
                classification = {}
        return jsonify({"run_id": run_id, "classification": classification, "raw_response": raw})

    # Extraction mode: parse rules array, save to DB
    extracted = _parse_llm_rules(raw)
    cl = content.lower()
    file_lines = content.split("\n")
    for rule in extracted:
        rt = rule.get("rule_text","")
        # NEVER trust the model's own line/char numbers — LLMs miscount lines (off by
        # one or more). ALWAYS derive position by locating the quoted text in the
        # document; if we can't locate it, leave position null rather than record a
        # wrong guess. Drop whatever the model reported up front.
        rule["char_start"] = rule["char_end"] = None
        rule["line_start"] = rule["line_end"] = None
        if not rt: continue
        loc = _locate_rule(content, cl, rt)
        if loc:
            cs, ce = loc   # don't shadow `b` (the request JSON) used later
            rule["char_start"] = cs
            rule["char_end"]   = ce
            rule["line_start"] = content[:cs].count("\n") + 1
            rule["line_end"]   = content[:ce].count("\n") + 1
        else:
            # Couldn't match the quote exactly — still derive the line from the doc
            # (word-overlap), never from the model. Find the line sharing the most words.
            words = [w for w in re.split(r'\W+', rt.lower()) if len(w) > 3]
            if words:
                best_li, best_score = -1, 0
                for li, line in enumerate(file_lines):
                    ll = line.lower()
                    score = sum(1 for w in words if w in ll)
                    if score > best_score:
                        best_score, best_li = score, li
                if best_li >= 0 and best_score >= max(2, len(words) // 3):
                    rule["line_start"] = best_li + 1
                    rule["line_end"]   = best_li + 1

    if annotator:
        con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
    con.execute(
        "INSERT INTO llm_runs(id,file_id,prompt,model,raw_response,rule_count) VALUES(?,?,?,?,?,?)",
        [run_id, fid, prompt, model, raw, len(extracted)]
    )
    # A judge run replaces this annotator's entire MACHINE set for the file (both
    # the blue 'llm' and green 'revise' layers) — extract and revise are alternative
    # machine passes, never stacked. Human ('hand') labels and other annotators are
    # left untouched.
    if b.get("replace_llm"):
        con.execute(
            "DELETE FROM extracted_rules WHERE file_id=? AND source IN ('llm','revise')"
            " AND annotator IS NOT DISTINCT FROM ?",
            [fid, annotator or None]
        )

    def _norm_tag(t):
        t = (t or "").strip().upper()
        return t if t in ("PROHIBITION", "PRESCRIPTION", "PERMISSION", "PREFERENCE") else None

    def _norm_power(tag, pt):
        if tag != "PREFERENCE":
            return None
        pt = (pt or "").strip().lower()
        return pt if pt in ("norm", "strategy") else None

    saved = []
    insert_rows = []   # collected and written in ONE batched statement below —
                       # per-row execute() is a network round-trip each on MotherDuck
    for rule in extracted:
        rt = rule.get("rule_text","").strip()
        if not rt: continue
        eid = make_id(fid, pass_source, annotator, run_id, rt)
        # Each item self-labels its kind; deontic tags apply to rules only.
        kind = "context" if (rule.get("kind") == "context") else "rule"
        tag = _norm_tag(rule.get("tag")) if kind == "rule" else None
        power_type = _norm_power(tag, rule.get("power_type"))
        rationale = rule.get("rationale") or rule.get("llm_rationale")
        insert_rows.append([eid, fid, rt,
              rule.get("char_start"), rule.get("char_end"),
              rule.get("line_start"), rule.get("line_end"),
              pass_source, run_id, model, tag, power_type, rationale, annotator or None, kind])
        saved.append({
            "id": eid, "rule_text": rt,
            "char_start": rule.get("char_start"), "char_end": rule.get("char_end"),
            "line_start": rule.get("line_start"), "line_end": rule.get("line_end"),
            "source": pass_source, "llm_run_id": run_id, "notes": None,
            "extracted_by": model, "annotator": annotator, "kind": kind,
            "tag": tag, "power_type": power_type, "llm_rationale": rationale,
            "context_type": None, "reviewed": False,
        })
    if insert_rows:
        # Single multi-row INSERT = one MotherDuck round-trip instead of N.
        ph = "(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?,?)"
        flat = [v for row in insert_rows for v in row]
        con.execute(
            "INSERT OR IGNORE INTO extracted_rules"
            " (id,file_id,rule_text,char_start,char_end,line_start,line_end,"
            "  source,llm_run_id,notes,extracted_by,tag,power_type,llm_rationale,annotator,kind)"
            " VALUES " + ",".join([ph] * len(insert_rows)),
            flat,
        )
    con.close()
    return jsonify({"run_id": run_id, "rules": saved, "raw_response": raw})

@app.route("/api/llm-runs/<run_id>", methods=["DELETE"])
def delete_llm_run(run_id):
    con = annot_con()
    con.execute("DELETE FROM extracted_rules WHERE llm_run_id=?", [run_id])
    con.execute("DELETE FROM llm_runs WHERE id=?", [run_id])
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Routes – multi-judge voting (Extract / Label / Relation)
# ---------------------------------------------------------------------------
_TAGS = ("PROHIBITION", "PRESCRIPTION", "PERMISSION", "PREFERENCE")

def _norm_tag(t):
    t = (t or "").strip().upper()
    return t if t in _TAGS else None

def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0: return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) // 2

def _overlap_frac(a, b):
    """Reciprocal char-overlap of two [start,end) spans (min of the two ratios)."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0: return 0.0
    la, lb = a[1] - a[0], b[1] - b[0]
    if la <= 0 or lb <= 0: return 0.0
    return min(inter / la, inter / lb)

def _cluster_spans(spans, thresh_frac=0.5):
    """Single-linkage cluster span proposals of the SAME kind whose char ranges
    reciprocally overlap ≥ thresh_frac. Returns a list of groups (lists of spans)."""
    n = len(spans)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if spans[i].get("kind") != spans[j].get("kind"): continue
            ai = (spans[i]["char_start"], spans[i]["char_end"])
            aj = (spans[j]["char_start"], spans[j]["char_end"])
            if _overlap_frac(ai, aj) >= thresh_frac:
                ra, rb = find(i), find(j)
                if ra != rb: parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(spans[i])
    return list(groups.values())

class JudgeError(Exception):
    """A user-facing judge failure (bad state / API error) surfaced via the job."""
    def __init__(self, msg, code=400):
        super().__init__(msg); self.msg = msg; self.code = code

# In-flight multi-judge jobs, so the loading UI can poll per-judge progress. Each
# judge is an opaque single LLM call, so a judge's bar is indeterminate while
# 'running' and fills on 'done'/'error' — letting the user watch them finish in
# parallel at their own pace.
_JUDGE_JOBS = {}
_JUDGE_LOCK = threading.Lock()

# Serializes multi-statement judge writes on the shared MotherDuck session so a
# concurrent reader (e.g. another viewer's GET /api/rules) never sees a half-applied
# DELETE+INSERT, and two writers can't conflict on the same tuples.
_DB_WRITE_LOCK = threading.Lock()

@contextlib.contextmanager
def _atomic(con):
    """Run a judge save under the write lock inside one transaction (commit on
    success, rollback on error) so it's atomic to other cursors on the session.
    Hold this ONLY around DB writes — never around the slow LLM fan-out."""
    with _DB_WRITE_LOCK:
        con.execute("BEGIN TRANSACTION")
        try:
            yield
            con.execute("COMMIT")
        except Exception:
            try: con.execute("ROLLBACK")
            except Exception: pass
            raise

def _cleanup_jobs():
    """Drop finished/stale jobs (called under _JUDGE_LOCK). Only reap a job that is
    done OR has gone quiet past the ceiling — never one whose thread is still
    actively reporting progress (its ts is refreshed by the progress callback)."""
    now = time.time()
    for k in [k for k, v in _JUDGE_JOBS.items()
              if (v.get("done") and now - v.get("ts", 0) > 60) or now - v.get("ts", 0) > 1800]:
        _JUDGE_JOBS.pop(k, None)

def _judge_fanout(judges, sys_msg, usr_msg, api_key, progress_cb=None):
    """Run N judges CONCURRENTLY (one Perplexity call each). Returns a list of
    (judge_idx, model, raw_text, error) preserving judge order. progress_cb(idx,
    status) fires as each judge completes ('done' / 'error') — used for the
    per-judge progress bars on the loading screen."""
    results = [None] * len(judges)
    def work(i):
        model = (judges[i].get("model") or "sonar").strip()
        try:
            raw = _perplexity_complete(model, sys_msg, usr_msg, api_key, max_out=32000)
            return i, model, raw, None
        except urllib.error.HTTPError as e:
            return i, model, "", f"API {e.code}: {e.read().decode()[:200]}"
        except Exception as e:
            return i, model, "", str(e)
    # max_workers == judge count (capped) so every judge in a normal panel runs at once.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, max(1, len(judges)))) as ex:
        futs = [ex.submit(work, i) for i in range(len(judges))]
        for fut in concurrent.futures.as_completed(futs):
            i, model, raw, err = fut.result()
            results[i] = (i, model, raw, err)
            if progress_cb:
                try: progress_cb(i, "error" if err else "done")
                except Exception: pass
    return results

def _save_run_and_votes(con, run_id, fid, annotator, mode, threshold, judges, votes):
    """Persist a judge_runs row + every raw per-judge vote, replacing this
    annotator's previous run/votes for the same (file, mode)."""
    con.execute("DELETE FROM judge_votes WHERE file_id=? AND mode=? AND annotator IS NOT DISTINCT FROM ?",
                [fid, mode, annotator or None])
    con.execute("DELETE FROM judge_runs WHERE file_id=? AND mode=? AND annotator IS NOT DISTINCT FROM ?",
                [fid, mode, annotator or None])
    con.execute("INSERT INTO judge_runs(id,file_id,annotator,mode,threshold,judge_total,models) VALUES(?,?,?,?,?,?,?)",
                [run_id, fid, annotator or None, mode, threshold, len(judges),
                 json.dumps([j.get("model") for j in judges])])
    # ONE batched multi-row INSERT instead of a network round-trip per vote — the
    # per-row version made the post-fanout "aggregating" wait take seconds on MotherDuck.
    if votes:
        ph = "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        flat = []
        for k, v in enumerate(votes):
            vid = make_id(run_id, v.get("judge_idx"), v.get("rule_id") or "", v.get("source_id") or "",
                          v.get("target_id") or "", v.get("char_start"), v.get("char_end"), v.get("value") or "", k)
            flat += [vid, run_id, fid, annotator or None, mode, v.get("judge_idx"), v.get("model"),
                     v.get("char_start"), v.get("char_end"), v.get("kind"), v.get("rule_id"),
                     v.get("source_id"), v.get("target_id"), v.get("value"), v.get("rationale")]
        con.execute(
            "INSERT OR IGNORE INTO judge_votes"
            " (id,run_id,file_id,annotator,mode,judge_idx,model,char_start,char_end,kind,rule_id,source_id,target_id,value,rationale)"
            " VALUES " + ",".join([ph] * len(votes)), flat)

def _judge_gate(mode, fid, annotator):
    """Synchronous pre-flight checks so the UI gets an immediate error (not via
    polling). Label needs ≥1 rule; Relation needs ≥2 entities AND every rule
    tagged (relations are built from context + tagged rules)."""
    if mode not in ("label", "relation"):
        return None
    con = annot_con()
    rows = con.execute(
        "SELECT kind, tag FROM extracted_rules WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?"
        " AND (accepted IS NULL OR accepted=TRUE)", [fid, annotator or None]).fetchall()
    con.close()
    rules = [r for r in rows if (r[0] or "rule") == "rule"]
    if mode == "label":
        if not rules:
            return "No rules to label — run Extract first."
        return None
    # relation
    if len(rows) < 2:
        return "Need at least 2 rules/context in this file to relate — run Extract first."
    untagged = sum(1 for r in rules if not r[1])
    if untagged:
        return (f"Tag every rule in Label first — {untagged} rule{'s' if untagged != 1 else ''} "
                "still untagged. (Relations are built from context + tagged rules.)")
    return None

@app.route("/api/judge", methods=["POST"])
def run_judge():
    """Kick off multi-judge voting for one stage. Judges run CONCURRENTLY in a
    background thread; this returns a run_id immediately and the client polls
    /api/judge-progress/<run_id> to watch each judge finish and collect the result.
    Aggregation: Extract = char-overlap clusters; Label = per-rule majority tag;
    Relation = per-edge majority type (over context + tagged rules)."""
    b = request.json or {}
    fid       = (b.get("file_id") or "").strip()
    mode      = (b.get("mode") or "extract").strip().lower()
    annotator = (b.get("annotator") or "").strip()
    prompt    = (b.get("prompt") or "").strip()
    judges    = b.get("judges") or []
    if isinstance(judges, list) and judges and isinstance(judges[0], str):
        judges = [{"model": m} for m in judges]
    if not judges:
        judges = [{"model": (b.get("model") or "sonar")}]
    judges = [{"model": (j.get("model") or "sonar").strip()} for j in judges]
    try: threshold = int(b.get("threshold") or 1)
    except Exception: threshold = 1
    threshold = max(1, min(threshold, len(judges)))
    if mode not in ("extract", "label", "relation"):
        return jsonify({"error": "mode must be extract|label|relation"}), 400
    if not fid or not prompt:
        return jsonify({"error": "file_id and prompt required"}), 400
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        return jsonify({"error": "PERPLEXITY_API_KEY not set"}), 400
    sc = sample_con()
    row = sc.execute("SELECT content FROM sample WHERE id=?", [fid]).fetchone()
    if not row: return jsonify({"error": "file not found"}), 404
    content = row[0] or ""
    gate = _judge_gate(mode, fid, annotator)
    if gate:
        return jsonify({"error": gate}), 400
    run_id = make_id(fid, mode, annotator, prompt, datetime.now(timezone.utc).isoformat())
    with _JUDGE_LOCK:
        _cleanup_jobs()
        _JUDGE_JOBS[run_id] = {
            "mode": mode, "judge_total": len(judges), "threshold": threshold,
            "judges": [{"model": j["model"], "status": "running"} for j in judges],
            "done": False, "result": None, "error": None, "ts": time.time(),
        }
    t = threading.Thread(target=_run_judge_job,
                         args=(run_id, mode, fid, annotator, prompt, judges, threshold, content, api_key),
                         daemon=True)
    t.start()
    return jsonify({"run_id": run_id, "started": True, "judge_total": len(judges),
                    "judges": [j["model"] for j in judges]})

def _run_judge_job(run_id, mode, fid, annotator, prompt, judges, threshold, content, api_key):
    """Background worker: run the stage's judges (with live per-judge progress),
    aggregate, and stash the result/error on the job for the client to poll."""
    def cb(idx, status):
        with _JUDGE_LOCK:
            job = _JUDGE_JOBS.get(run_id)
            if job and 0 <= idx < len(job["judges"]):
                job["judges"][idx]["status"] = status
                job["ts"] = time.time()   # keep the job fresh so _cleanup_jobs won't reap it mid-run
    fn = {"extract": _judge_extract, "label": _judge_label, "relation": _judge_relation}[mode]
    try:
        payload = fn(fid, annotator, prompt, judges, threshold, content, api_key, run_id, cb)
        with _JUDGE_LOCK:
            job = _JUDGE_JOBS.get(run_id)
            if job: job["result"] = payload; job["done"] = True; job["ts"] = time.time()
    except JudgeError as je:
        with _JUDGE_LOCK:
            job = _JUDGE_JOBS.get(run_id)
            if job: job["error"] = je.msg; job["done"] = True; job["ts"] = time.time()
    except Exception as e:
        with _JUDGE_LOCK:
            job = _JUDGE_JOBS.get(run_id)
            if job: job["error"] = str(e); job["done"] = True; job["ts"] = time.time()

@app.route("/api/judge-progress/<run_id>")
def judge_progress(run_id):
    """Poll a running/finished multi-judge job. While running, returns each judge's
    status; once done, returns the aggregation result (then drops the job)."""
    with _JUDGE_LOCK:
        job = _JUDGE_JOBS.get(run_id)
        if not job:
            return jsonify({"error": "unknown or expired run", "missing": True}), 404
        out = {
            "run_id": run_id, "mode": job["mode"], "judge_total": job["judge_total"],
            "threshold": job["threshold"], "judges": [dict(j) for j in job["judges"]],
            "done": job["done"], "error": job["error"],
            "result": job["result"] if job["done"] else None,
        }
        if job["done"]:
            _JUDGE_JOBS.pop(run_id, None)   # one final read, then free it
    return jsonify(out)

def _judge_extract(fid, annotator, prompt, judges, threshold, content, api_key, run_id, progress_cb=None):
    sys_msg = ("You identify spans in a document and return ONLY a single valid JSON array — "
               "no prose, no markdown fences. Use exactly the field schema in the user's instructions.")
    usr_msg = f"{prompt}\n\nDocument:\n---\n{content[:30000]}\n---\n\nReturn ONLY a JSON array."
    fan = _judge_fanout(judges, sys_msg, usr_msg, api_key, progress_cb)
    errors = [e for (_, _, _, e) in fan if e]
    cl = content.lower()
    votes, span_props = [], []
    for idx, model, raw, err in fan:
        if err: continue
        for item in _parse_llm_rules(raw):
            rt = (item.get("rule_text") or "").strip()
            if not rt: continue
            loc = _locate_rule(content, cl, rt)
            if not loc: continue
            cs, ce = loc
            kind = "context" if (item.get("kind") == "context") else "rule"
            rationale = item.get("rationale") or item.get("llm_rationale")
            votes.append({"judge_idx": idx, "model": model, "char_start": cs, "char_end": ce,
                          "kind": kind, "value": kind, "rationale": rationale})
            span_props.append({"judge_idx": idx, "char_start": cs, "char_end": ce,
                               "kind": kind, "rationale": rationale})
    if not votes and errors:
        raise JudgeError(errors[0], 502)
    clusters = _cluster_spans(span_props)
    # Build the accepted rows first (pure work, no DB) so the write lock is held only
    # for the actual save below.
    accepted, insert_rows = 0, []
    for grp in clusters:
        nj = len({s["judge_idx"] for s in grp})
        if nj < threshold: continue          # sub-threshold → candidate (light), not a row
        cs, ce = _median([s["char_start"] for s in grp]), _median([s["char_end"] for s in grp])
        rt = content[cs:ce].strip()
        if not rt: continue
        kinds = [s["kind"] for s in grp]
        kind = "context" if kinds.count("context") > kinds.count("rule") else "rule"
        ls = content[:cs].count("\n") + 1
        le = content[:ce].count("\n") + 1
        rationale = next((s.get("rationale") for s in grp if s.get("rationale")), None)
        eid = make_id(fid, "extract", annotator, cs, ce)   # stable per span → relations survive re-runs
        insert_rows.append([eid, fid, rt, cs, ce, ls, le, "llm", run_id, None, "judge",
                            None, kind, rationale, annotator or None, True, nj, len(judges)])
        accepted += 1
    con = annot_con()
    try:
        with _atomic(con):
            if annotator:
                con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
            _save_run_and_votes(con, run_id, fid, annotator, "extract", threshold, judges, votes)
            # An Extract run replaces this annotator's previous machine rules (keeps hand).
            con.execute("DELETE FROM extracted_rules WHERE file_id=? AND source IN ('llm','revise')"
                        " AND annotator IS NOT DISTINCT FROM ?", [fid, annotator or None])
            if insert_rows:
                ph = "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                flat = [v for r in insert_rows for v in r]
                con.execute(
                    "INSERT OR IGNORE INTO extracted_rules"
                    " (id,file_id,rule_text,char_start,char_end,line_start,line_end,source,llm_run_id,"
                    "  notes,extracted_by,tag,kind,llm_rationale,annotator,accepted,vote_count,judge_total)"
                    " VALUES " + ",".join([ph] * len(insert_rows)), flat)
    finally:
        con.close()
    return {"run_id": run_id, "mode": "extract", "accepted": accepted,
            "judge_total": len(judges), "threshold": threshold, "errors": errors}

def _judge_label(fid, annotator, prompt, judges, threshold, content, api_key, run_id, progress_cb=None):
    con = annot_con()
    rows = con.execute(
        "SELECT id, kind, rule_text, line_start FROM extracted_rules"
        " WHERE file_id=? AND annotator IS NOT DISTINCT FROM ? AND (accepted IS NULL OR accepted=TRUE)"
        " ORDER BY COALESCE(line_start,999999), COALESCE(char_start,999999)",
        [fid, annotator or None]).fetchall()
    targets = [r for r in rows if (r[1] or "rule") == "rule"]
    if not targets:
        con.close()
        raise JudgeError("No rules to label — run Extract first (or add rules).")
    label_to_id, lines = {}, []
    for i, (rid, kind, text, ls) in enumerate(targets, 1):
        lab = f"R{i}"; label_to_id[lab] = rid
        lines.append(f'{lab} (line {ls}): {(text or "")[:200].strip()}')
    node_block = "\n".join(lines)
    sys_msg = ("You assign exactly one deontic tag to each listed rule and return ONLY a valid "
               "JSON array — no prose, no fences. Use exactly the field schema in the user's instructions.")
    usr_msg = (f"{prompt}\n\nRULES to tag (reference each by its R-label):\n{node_block}\n\n"
               f"Document:\n---\n{content[:25000]}\n---\n\nReturn ONLY a JSON array.")
    fan = _judge_fanout(judges, sys_msg, usr_msg, api_key, progress_cb)
    errors = [e for (_, _, _, e) in fan if e]
    votes, tally = [], {}
    for idx, model, raw, err in fan:
        if err: continue
        for item in _parse_llm_rules(raw):
            lab = str(item.get("rule") or item.get("rule_ref") or item.get("ref")
                      or item.get("R") or item.get("id") or "").strip()
            rid = label_to_id.get(lab)
            if not rid: continue
            tag = _norm_tag(item.get("tag") or item.get("value") or item.get("label"))
            rationale = item.get("rationale") or item.get("llm_rationale")
            votes.append({"judge_idx": idx, "model": model, "rule_id": rid,
                          "value": tag, "rationale": rationale})
            if tag:
                tally.setdefault(rid, {}).setdefault(tag, 0)
                tally[rid][tag] += 1
    if not votes and errors:
        con.close(); raise JudgeError(errors[0], 502)
    labeled = 0
    try:
        with _atomic(con):
            if annotator:
                con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
            _save_run_and_votes(con, run_id, fid, annotator, "label", threshold, judges, votes)
            for rid, counts in tally.items():
                best_n = max(counts.values())
                winners = [t for t, c in counts.items() if c == best_n]
                # record ring counts regardless; only APPLY the tag on a clear ≥threshold
                # win. Store the LLM's pick in BOTH tag (effective) and llm_tag (preserved
                # so a later human change can be detected/reverted).
                if len(winners) == 1 and best_n >= threshold:
                    con.execute("UPDATE extracted_rules SET tag=?, llm_tag=?, llm_run_id=?, vote_count=?, judge_total=? WHERE id=?",
                                [winners[0], winners[0], run_id, best_n, len(judges), rid])
                    labeled += 1
                else:
                    con.execute("UPDATE extracted_rules SET llm_run_id=?, vote_count=?, judge_total=? WHERE id=?",
                                [run_id, best_n, len(judges), rid])
    finally:
        con.close()
    return {"run_id": run_id, "mode": "label", "labeled": labeled,
            "judge_total": len(judges), "threshold": threshold, "errors": errors}

def _judge_relation(fid, annotator, prompt, judges, threshold, content, api_key, run_id, progress_cb=None):
    con = annot_con()
    nodes = con.execute(
        "SELECT id, kind, tag, rule_text, line_start FROM extracted_rules"
        " WHERE file_id=? AND annotator IS NOT DISTINCT FROM ? AND (accepted IS NULL OR accepted=TRUE)"
        " ORDER BY COALESCE(line_start,999999), COALESCE(char_start,999999)",
        [fid, annotator or None]).fetchall()
    if len(nodes) < 2:
        con.close()
        raise JudgeError("Need at least 2 rules/context in this file to relate.")
    # Present each entity WITH its deontic tag (rules) or CONTEXT, so the judges
    # relate based on both the context spans and the rules' tags.
    label_to_id, lines = {}, []
    for i, (nid, kind, tag, text, ls) in enumerate(nodes, 1):
        lab = f"R{i}"; label_to_id[lab] = nid
        kindlab = "context" if (kind == "context") else ("rule/" + (tag or "UNTAGGED"))
        lines.append(f'{lab} [{kindlab}] (line {ls}): {(text or "")[:200].strip()}')
    node_block = "\n".join(lines)
    # PROJECT CONTEXT — relations must reflect how the rules actually interact in THIS
    # project, not just surface wording. Feed the repo identity so a web-capable judge
    # can consult the real repository when it helps. (Repo metadata lives in `sample`,
    # keyed by the rule's file_id.)
    proj = ""
    try:
        mrow = sample_con().execute(
            "SELECT repo_name, source_url, file_type FROM sample WHERE id=?", [fid]).fetchone()
    except Exception:
        mrow = None
    if mrow:
        repo, url, ftype = mrow[0], mrow[1], mrow[2]
        bits = []
        if repo:  bits.append(f"repository **{repo}**")
        if ftype: bits.append(f"config file type `{ftype}`")
        proj = ("\n\nPROJECT CONTEXT — these rules come from the " + (", ".join(bits) or "project")
                + (f" (source: {url})" if url else "") + ". Ground every relation in what THIS project "
                "actually is and how its rules interact in practice — not just surface textual similarity. "
                "If it would help and you are able to, consult the actual repository"
                + (f" at {url}" if url else "") + " (its README, structure, and related files) to understand "
                "the project's purpose before deciding how the rules relate.")
    sys_msg = ("You propose directed relations between the listed entities and return ONLY a single "
               "valid JSON array — no prose, no fences. Use exactly the field schema in the user's instructions. "
               "Judge relations by the project's real semantics, not just wording; consult the repository when useful.")
    usr_msg = (f"{prompt}{proj}\n\nENTITIES in this document — each is either a CONTEXT span or a rule with its "
               f"deontic tag (PROHIBITION/PRESCRIPTION/PERMISSION/PREFERENCE). Use BOTH the tags and the "
               f"context when deciding how they relate. Reference them by their R-label:\n{node_block}\n\n"
               f"Document:\n---\n{content[:25000]}\n---\n\nReturn ONLY a JSON array of relations.")
    fan = _judge_fanout(judges, sys_msg, usr_msg, api_key, progress_cb)
    errors = [e for (_, _, _, e) in fan if e]
    # Tally agreement on the (source, target, TYPE) — a judge counts toward an edge
    # only for the type it actually voted, so the acceptance threshold and the stored
    # vote_count reflect agreement on the relation TYPE, not just "some edge here".
    # One judge's first vote for a given (pair, type) is what counts (no double-count).
    votes = []
    edge_type_judges = {}    # (src,tgt,type) -> set(judge_idx)
    edge_type_rat = {}       # (src,tgt,type) -> a representative rationale
    for idx, model, raw, err in fan:
        if err: continue
        for e in _parse_llm_rules(raw):
            s_lab = str(e.get("source") or e.get("source_id") or "").strip()
            t_lab = str(e.get("target") or e.get("target_id") or "").strip()
            src, tgt = label_to_id.get(s_lab), label_to_id.get(t_lab)
            if not src or not tgt or src == tgt: continue
            rtype = (e.get("type") or e.get("relation_type") or "").strip().lower()
            if rtype not in _REL_TYPES:
                rtype = "define" if rtype.startswith(("support", "define")) else None
            rationale = e.get("rationale") or e.get("llm_rationale")
            votes.append({"judge_idx": idx, "model": model, "source_id": src, "target_id": tgt,
                          "value": rtype, "rationale": rationale})
            if rtype:   # only typed votes contribute to an acceptable edge
                edge_type_judges.setdefault((src, tgt, rtype), set()).add(idx)
                if rationale and (src, tgt, rtype) not in edge_type_rat:
                    edge_type_rat[(src, tgt, rtype)] = rationale
    if not votes and errors:
        con.close(); raise JudgeError(errors[0], 502)
    # Per ordered pair, pick the type with the most distinct-judge support (plurality);
    # accept ONE majority edge per pair when that winning type reaches the threshold.
    pair_best = {}           # (src,tgt) -> (type, n_judges, rationale)
    for (src, tgt, rtype), jset in edge_type_judges.items():
        n = len(jset)
        cur = pair_best.get((src, tgt))
        if cur is None or n > cur[1]:
            pair_best[(src, tgt)] = (rtype, n, edge_type_rat.get((src, tgt, rtype)))
    accepted = 0
    try:
        with _atomic(con):
            if annotator:
                con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [annotator])
            _save_run_and_votes(con, run_id, fid, annotator, "relation", threshold, judges, votes)
            con.execute("DELETE FROM relations WHERE file_id=? AND source='llm' AND annotator IS NOT DISTINCT FROM ?",
                        [fid, annotator or None])
            for (src, tgt), (rtype, nj, rat) in pair_best.items():
                if nj < threshold: continue          # winning type didn't reach agreement
                rid = make_id(fid, src, tgt, "llm", annotator)   # one majority llm edge per pair
                con.execute(
                    "INSERT OR IGNORE INTO relations"
                    " (id,file_id,source_id,target_id,relation_type,notes,source,llm_run_id,llm_rationale,"
                    "  annotator,llm_relation_type,accepted,vote_count,judge_total)"
                    " VALUES (?,?,?,?,NULL,NULL,'llm',?,?,?,?,TRUE,?,?)",
                    [rid, fid, src, tgt, run_id, rat, annotator or None, rtype, nj, len(judges)])
                accepted += 1
    finally:
        con.close()
    return {"run_id": run_id, "mode": "relation", "accepted": accepted,
            "judge_total": len(judges), "threshold": threshold, "errors": errors}

@app.route("/api/judge-data/<fid>")
def judge_data(fid):
    """Latest multi-judge run + every per-judge vote for each mode — powers the
    vote rings, the inspection panel, and Extract candidate/light highlights."""
    annotator = (request.args.get("annotator") or "").strip()
    con = annot_con()
    out = {}
    for mode in ("extract", "label", "relation"):
        run = con.execute(
            "SELECT id, threshold, judge_total, models FROM judge_runs"
            " WHERE file_id=? AND mode=? AND annotator IS NOT DISTINCT FROM ?"
            " ORDER BY created_at DESC LIMIT 1", [fid, mode, annotator or None]).fetchone()
        if not run:
            out[mode] = None; continue
        run_id, threshold, judge_total, models = run
        vrows = con.execute(
            "SELECT judge_idx, model, char_start, char_end, kind, rule_id, source_id, target_id, value, rationale"
            " FROM judge_votes WHERE run_id=? ORDER BY judge_idx", [run_id]).fetchall()
        votes = [{"judge_idx": v[0], "model": v[1], "char_start": v[2], "char_end": v[3],
                  "kind": v[4], "rule_id": v[5], "source_id": v[6], "target_id": v[7],
                  "value": v[8], "rationale": v[9]} for v in vrows]
        info = {"run_id": run_id, "threshold": threshold, "judge_total": judge_total,
                "models": json.loads(models) if models else [], "votes": votes}
        if mode == "extract":
            spans = [{"judge_idx": v["judge_idx"], "char_start": v["char_start"],
                      "char_end": v["char_end"], "kind": v["kind"]}
                     for v in votes if v["char_start"] is not None and v["char_end"] is not None]
            cands = []
            for grp in _cluster_spans(spans):
                nj = len({s["judge_idx"] for s in grp})
                if nj >= (threshold or 1): continue   # accepted clusters are already rules
                cs, ce = _median([s["char_start"] for s in grp]), _median([s["char_end"] for s in grp])
                kinds = [s["kind"] for s in grp]
                kind = "context" if kinds.count("context") > kinds.count("rule") else "rule"
                cands.append({"char_start": cs, "char_end": ce, "kind": kind, "vote_count": nj})
            info["candidates"] = cands
        out[mode] = info
    con.close()
    return jsonify(out)

# ---------------------------------------------------------------------------
# Routes – CSV export page
# ---------------------------------------------------------------------------
@app.route("/export")
def export_page():
    annotator = (request.args.get("annotator") or "").strip()
    file_id   = (request.args.get("file_id") or "").strip()   # optional: limit to one file
    acon = annot_con()
    # ── Rules / context entities ─────────────────────────────────────────
    rule_sql = """
        SELECT r.id, r.file_id, r.rule_text, r.kind, r.tag,
               r.line_start, r.line_end, r.char_start, r.char_end,
               r.source, r.extracted_by, r.notes, r.llm_rationale,
               r.annotator, r.created_at, r.context_type
        FROM extracted_rules r
    """
    rconds, rparams = [], []
    if annotator: rconds.append("r.annotator=?"); rparams.append(annotator)
    if file_id:   rconds.append("r.file_id=?");   rparams.append(file_id)
    if rconds: rule_sql += " WHERE " + " AND ".join(rconds)
    rule_sql += " ORDER BY r.file_id, COALESCE(r.line_start, 999999), r.created_at"
    rule_rows = acon.execute(rule_sql, rparams).fetchall()

    # ── Relations (directed edges between the entities above) ─────────────
    rel_sql = """
        SELECT e.id, e.file_id, e.source_id, e.target_id,
               e.relation_type, e.llm_relation_type, e.source,
               e.notes, e.llm_rationale, e.annotator, e.created_at
        FROM relations e
    """
    econds, eparams = [], []
    if annotator: econds.append("e.annotator=?"); eparams.append(annotator)
    if file_id:   econds.append("e.file_id=?");   eparams.append(file_id)
    if econds: rel_sql += " WHERE " + " AND ".join(econds)
    rel_sql += " ORDER BY e.file_id, e.created_at"
    rel_rows = acon.execute(rel_sql, eparams).fetchall()
    acon.close()

    # id → rule_text so a relation row can show its endpoints' text, not just ids
    rule_text_map = {r[0]: r[2] for r in rule_rows}

    sc = sample_con()
    url_map = {}
    fids = list({r[1] for r in rule_rows} | {e[1] for e in rel_rows})
    if fids:
        placeholders = ",".join("?" * len(fids))
        url_map = {row[0]: row[1] for row in sc.execute(
            f"SELECT id, source_url FROM sample WHERE id IN ({placeholders})", fids
        ).fetchall()}
    # NOTE: sample_con() is a cached, shared read-only connection — do NOT close it.

    import csv, io
    blank = lambda v: "" if v is None else v
    buf = io.StringIO()
    writer = csv.writer(buf)
    # One flat CSV for both entity types — filter on `record_type` downstream.
    # rule/context rows fill the rule_* columns; relation rows fill the rel_* ones.
    writer.writerow([
        "record_type","annotator","source_url","source","extracted_by",
        "kind","tag","context_type","rule_text","line_start","line_end","char_start","char_end",
        "relation_type","llm_relation_type","source_rule_text","target_rule_text",
        "user_comment","llm_rationale","id","source_id","target_id","created_at",
    ])
    for r in rule_rows:
        writer.writerow([
            r[3] or "rule",                                  # record_type (= kind)
            r[13] or "", url_map.get(r[1],""), r[9] or "", r[10] or "",
            r[3] or "rule", r[4] or "", r[15] or "", r[2],   # kind, tag, context_type, rule_text
            blank(r[5]), blank(r[6]), blank(r[7]), blank(r[8]),  # line/char start+end
            "", "", "", "",                                  # relation-only columns
            r[11] or "", r[12] or "",                        # user_comment, llm_rationale
            r[0], "", "",                                    # id, (no source/target)
            str(r[14] or ""),
        ])
    for e in rel_rows:
        writer.writerow([
            "relation",
            e[9] or "", url_map.get(e[1],""), e[6] or "", "",   # annotator, url, source(hand/llm), extracted_by
            "", "", "", "",                                  # kind, tag, context_type, rule_text (n/a)
            "", "", "", "",                                  # line/char (n/a)
            e[4] or "", e[5] or "",                          # relation_type (user), llm_relation_type
            rule_text_map.get(e[2], f"(rule {e[2]})"),       # source_rule_text
            rule_text_map.get(e[3], f"(rule {e[3]})"),       # target_rule_text
            e[7] or "", e[8] or "",                          # user_comment (notes), llm_rationale
            e[0], e[2], e[3],                                # id, source_id, target_id
            str(e[10] or ""),
        ])
    csv_text = buf.getvalue()
    row_count = len(rule_rows)
    rel_count = len(rel_rows)

    # Human-readable scope for the header + a matching download filename.
    if file_id:
        repo = None
        if sc:
            frow = sc.execute("SELECT repo_name FROM sample WHERE id=?", [file_id]).fetchone()
            repo = frow[0] if frow else None
        scope_label = f"current file — {repo or file_id}"
        dl_name = f"rules_{(repo or file_id).replace('/', '_')}.csv"
    else:
        scope_label = "all labeled files"
        dl_name = "rules_export_all.csv"
    if annotator:
        scope_label += f" · annotator: {annotator}"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Rules &amp; Relations Export</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f8f8fc; }}
  .bar {{ display:flex; align-items:center; gap:12px; padding:12px 18px;
          background:#fff; border-bottom:1px solid #e0e0ee; }}
  .bar h2 {{ margin:0; font-size:15px; color:#333; flex:1; }}
  .bar small {{ color:#888; font-size:12px; }}
  button {{ padding:6px 16px; border-radius:20px; border:none; cursor:pointer;
            font-size:13px; font-weight:600; }}
  .dl  {{ background:#6366f1; color:#fff; }}
  .dl:hover {{ background:#4f46e5; }}
  .cp  {{ background:#f0f0f8; color:#4338ca; border:1px solid #c7d2fe; }}
  .cp:hover {{ background:#eef2ff; }}
  textarea {{ display:block; width:100%; height:calc(100vh - 60px);
              border:none; padding:16px 18px; font-family:monospace;
              font-size:12px; line-height:1.5; background:#f8f8fc;
              color:#222; resize:none; box-sizing:border-box; outline:none; }}
</style>
</head><body>
<div class="bar">
  <h2>Rules &amp; Relations Export</h2>
  <small>{scope_label} · {row_count} rule{"s" if row_count!=1 else ""} · {rel_count} relation{"s" if rel_count!=1 else ""}</small>
  <button class="cp" onclick="copyCSV()">Copy</button>
  <button class="dl" onclick="downloadCSV()">Download CSV</button>
</div>
<textarea id="csv" readonly>{csv_text.replace("&","&amp;").replace("<","&lt;")}</textarea>
<script>
const csv = document.getElementById('csv');
function copyCSV() {{
  csv.select(); document.execCommand('copy');
  const b = document.querySelector('.cp');
  b.textContent = 'Copied!';
  setTimeout(() => b.textContent = 'Copy', 1500);
}}
function downloadCSV() {{
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv.value);
  a.download = '{dl_name}'; a.click();
}}
csv.addEventListener('focus', () => csv.select());
</script>
</body></html>"""
    return html

# ---------------------------------------------------------------------------
# Routes – saved prompts
# ---------------------------------------------------------------------------
@app.route("/api/prompts")
def list_prompts():
    con = annot_con()
    rows = con.execute(
        "SELECT id, prompt, label, saved_at FROM saved_prompts ORDER BY saved_at DESC"
    ).fetchall()
    con.close()
    return jsonify([{"id":r[0],"prompt":r[1],"label":r[2],"saved_at":str(r[3])} for r in rows])

@app.route("/api/prompts", methods=["POST"])
def save_prompt():
    b = request.json or {}
    prompt = (b.get("prompt") or "").strip()
    label  = (b.get("label")  or "").strip() or None
    if not prompt: return jsonify({"error": "prompt required"}), 400
    pid = make_id(prompt, datetime.now(timezone.utc).isoformat())
    con = annot_con()
    con.execute(
        "INSERT INTO saved_prompts(id,prompt,label) VALUES(?,?,?)", [pid, prompt, label]
    )
    con.close()
    return jsonify({"id": pid, "prompt": prompt, "label": label})

@app.route("/api/prompts/<pid>", methods=["DELETE"])
def delete_prompt(pid):
    con = annot_con()
    con.execute("DELETE FROM saved_prompts WHERE id=?", [pid])
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Routes – annotators (the people doing the labeling)
# ---------------------------------------------------------------------------
@app.route("/api/annotators")
def list_annotators():
    con = annot_con()
    # union of registered annotators and any that appear on rules
    rows = con.execute("""
        SELECT name FROM annotators
        UNION
        SELECT DISTINCT annotator FROM extracted_rules WHERE annotator IS NOT NULL AND annotator <> ''
        ORDER BY 1
    """).fetchall()
    con.close()
    return jsonify([r[0] for r in rows])

@app.route("/api/annotators", methods=["POST"])
def create_annotator():
    b = request.json or {}
    name = (b.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    con = annot_con()
    con.execute("INSERT INTO annotators(name) VALUES(?) ON CONFLICT(name) DO NOTHING", [name])
    con.close()
    return jsonify({"ok": True, "name": name})

# ---------------------------------------------------------------------------
# Routes – app settings (persisted, editable judge/extract prompts + models)
# ---------------------------------------------------------------------------
# The judge/extract/relation prompts are PER-ANNOTATOR: stored as "<key>::<annotator>".
# Everything else (e.g. judge_model) stays global. An annotator with no saved value
# falls back to the built-in default the frontend already holds.
# Per-annotator, resettable settings. The three judge_cfg_* hold a JSON blob per
# mode: {threshold, judges:[{model}], prompt}. The legacy prompt keys are kept so
# old per-annotator prompts still resolve (and Restore-default still works).
PROMPT_KEYS = ("llm_judge_prompt", "llm_judge_prompt_revise", "llm_relation_prompt",
               "judge_cfg_extract", "judge_cfg_label", "judge_cfg_relation")

@app.route("/api/settings")
def get_settings():
    annotator = (request.args.get("annotator") or "").strip()
    con = annot_con()
    rows = con.execute("SELECT key, value FROM app_settings").fetchall()
    con.close()
    allset = {k: v for k, v in rows}
    out = {}
    for k, v in allset.items():
        if "::" in k or k in PROMPT_KEYS:
            continue          # per-annotator overrides + legacy-global prompts: resolved below
        out[k] = v            # global, non-prompt settings (e.g. judge_model)
    if annotator:             # only THIS annotator's customised prompts (else the frontend default wins)
        for pk in PROMPT_KEYS:
            v = allset.get(f"{pk}::{annotator}")
            if v is not None:
                out[pk] = v
    return jsonify(out)

@app.route("/api/settings", methods=["POST"])
def set_settings():
    b = request.json or {}
    annotator = (b.get("annotator") or "").strip()
    con = annot_con()
    for k, v in b.items():
        if k == "annotator":
            continue
        key = f"{k}::{annotator}" if (k in PROMPT_KEYS and annotator) else k
        con.execute(
            "INSERT INTO app_settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [key, v]
        )
    con.close()
    return jsonify({"ok": True})

@app.route("/api/settings/<key>", methods=["DELETE"])
def reset_setting(key):
    """Restore a prompt to its built-in default by dropping this annotator's override."""
    annotator = (request.args.get("annotator") or "").strip()
    full = f"{key}::{annotator}" if (key in PROMPT_KEYS and annotator) else key
    con = annot_con()
    con.execute("DELETE FROM app_settings WHERE key=?", [full])
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _locate_rule(content, cl, rt):
    """Return (char_start, char_end) for a rule quote, or None.
    Falls back to the longest matching prefix so quotes that span a paragraph
    break (e.g. '...respond like this: "Heads up') still anchor to the right
    spot instead of trusting the model's approximate line number."""
    rt = (rt or "").strip()
    if not rt:
        return None
    # 1. exact / case-insensitive full match
    idx = content.find(rt)
    if idx == -1:
        idx = cl.find(rt.lower())
    if idx >= 0:
        return idx, idx + len(rt)
    # 2. longest contiguous prefix that appears in the file (binary search —
    #    prefix membership is monotonic, so a longer found prefix implies the
    #    shorter ones are found too)
    rl = rt.lower()
    lo, hi, best_len, best_idx = 16, len(rl), -1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        j = cl.find(rl[:mid])
        if j != -1:
            best_len, best_idx = mid, j
            lo = mid + 1
        else:
            hi = mid - 1
    if best_idx >= 0 and best_len >= 20:
        return best_idx, best_idx + best_len
    return None

def _recover_json_objects(text):
    """Pull every complete top-level {...} object out of text, respecting
    strings/escapes. Recovers rules from a truncated JSON array (the cut-off
    final object is simply skipped)."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:        esc = False
            elif ch == '\\': esc = True
            elif ch == '"':  in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1]); start = None
    return objs

def _parse_llm_rules(text):
    text = re.sub(r'^```(?:json)?\s*','',text.strip())
    text = re.sub(r'\s*```$','',text).strip()
    # 1. clean whole-array parse
    try:
        d = json.loads(text)
        if isinstance(d, list): return d
    except: pass
    # 2. outermost [...] block
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, list): return d
        except: pass
    # 3. recover complete objects (handles truncated/streamed JSON — keeps every
    #    finished rule with its tag/rationale, drops only the cut-off last one)
    recovered = []
    for chunk in _recover_json_objects(text):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and obj.get("rule_text"):
                recovered.append(obj)
        except: pass
    if recovered:
        return recovered
    # 4. last resort: treat non-trivial lines as bare rule text
    return [{"rule_text": re.sub(r'^[-*\d.]+\s*','',l.strip())}
            for l in text.split('\n') if len(l.strip()) > 20]

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rule Annotator</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root { --insp-panel-width: 420px; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f0f6; height: 100vh; overflow: hidden; color: #1a1a2e;
}
body.resizing { cursor: col-resize; user-select: none; }
.layout { display: flex; height: 100vh; }

/* ── File panel ── */
.file-panel {
  width: 230px; flex-shrink: 0; background: #18182e;
  display: flex; flex-direction: column; overflow: hidden;
  border-right: 1px solid #2a2a4a;
}
.file-panel-head {
  padding: 12px 12px 10px; color: #fff; font-size: 13px; font-weight: 700;
  border-bottom: 1px solid #2a2a4a; flex-shrink: 0; display: flex;
  align-items: baseline; gap: 6px;
}
.file-panel-head small { font-weight: 400; font-size: 11px; color: #5a5a7a; }
.file-search { padding: 7px 10px; flex-shrink: 0; border-bottom: 1px solid #2a2a4a; }
.file-search input {
  width: 100%; padding: 5px 8px; border-radius: 6px; border: 1px solid #2a2a4a;
  background: #0f0f22; color: #d0d0f0; font-size: 12px;
}
.file-search input:focus { outline: none; border-color: #5050a0; }
.file-search input::placeholder { color: #4a4a6a; }
.file-list { flex: 1; overflow-y: auto; padding: 5px 0; }
.file-item {
  padding: 6px 10px 6px 8px; cursor: pointer; border-left: 3px solid transparent;
  transition: background 0.1s; display: flex; gap: 7px; align-items: flex-start;
}
.file-item:hover  { background: #22223a; }
.file-item.active { background: #272750; border-left-color: #6366f1; }
.file-idx { font-size: 10px; color: #4a4a6a; min-width: 22px; text-align: right; padding-top: 1px; flex-shrink: 0; font-variant-numeric: tabular-nums; }
.file-item.active .file-idx { color: #7070b0; }
.file-info { flex: 1; min-width: 0; }
.file-name { font-size: 12px; color: #c0c0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file-item.active .file-name { color: #fff; }
.file-meta { font-size: 10px; color: #5a5a7a; margin-top: 2px; display: flex; gap: 5px; align-items: center; flex-wrap: wrap; }
.badge { display: inline-flex; align-items: center; gap: 2px; padding: 1px 5px; border-radius: 10px; font-size: 10px; font-weight: 600; }
/* colors tuned for the dark file sidebar: red = human labels, blue = LLM labels
   (matches the document highlight colours: human red, machine blue) */
.badge.hand { background: rgba(239,68,68,0.22);  color: #f87171; }
.badge.llm  { background: rgba(59,130,246,0.26); color: #60a5fa; }
/* machine badge turns green once a "revise" pass has run on the file */
.badge.revised { background: rgba(34,197,94,0.24); color: #4ade80; }

/* ── Viewer ── */
.viewer-panel {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  background: #fff; min-width: 340px;
}
.viewer-head {
  padding: 9px 14px; background: #fafafa; border-bottom: 1px solid #eaeaf0;
  flex-shrink: 0; display: flex; align-items: center; gap: 10px; min-height: 40px;
}
.viewer-title { font-size: 13px; font-weight: 600; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }
.viewer-head a { font-size: 11px; color: #6366f1; text-decoration: none; flex-shrink: 0; }
.viewer-head a:hover { text-decoration: underline; }
.sel-bar {
  padding: 5px 14px; background: #f4f4fc; border-bottom: 1px solid #e4e4f0;
  display: flex; align-items: center; gap: 10px; flex-shrink: 0; min-height: 34px;
}
.sel-info { font-size: 11px; color: #7a7aaa; flex: 1; }
.sel-info .kb { color: #aaa; font-size: 10px; }
kbd {
  display: inline-block; padding: 1px 4px; border-radius: 3px;
  border: 1px solid #ccc; background: #f8f8f8; font-size: 10px;
  font-family: inherit; color: #555; line-height: 1.4;
}
.add-btn {
  padding: 3px 12px; border-radius: 20px; border: 1.5px solid #f59e0b;
  color: #f59e0b; background: transparent; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: all 0.12s; white-space: nowrap;
}
.add-btn:hover:not(:disabled) { background: #f59e0b; color: #fff; }
.add-btn:disabled { opacity: 0.32; cursor: default; }
.judge-btn {
  padding: 3px 12px; border-radius: 20px; border: 1.5px solid #6366f1;
  color: #6366f1; background: transparent; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: all 0.12s; white-space: nowrap;
}
.judge-btn:hover { background: #6366f1; color: #fff; }
.judge-btn.active { background: #6366f1; color: #fff; }
.comment-btn {
  padding: 3px 12px; border-radius: 20px; border: 1.5px solid #10b981;
  color: #059669; background: transparent; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: all 0.12s; white-space: nowrap;
}
.comment-btn:hover { background: #10b981; color: #fff; }
.comment-btn.active { background: #10b981; color: #fff; }
/* filled style when this file already has a comment from the current annotator */
.comment-btn.has-comment { background: #d1fae5; }
.comment-btn.has-comment:hover, .comment-btn.has-comment.active { background: #10b981; color: #fff; }
.viewer-body {
  flex: 1; display: flex; overflow: auto;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 13px; line-height: 1.6;
}
.line-nums {
  white-space: normal; padding: 14px 10px 14px 12px; color: #bbbbd0;
  text-align: right; user-select: none; border-right: 1px solid #f0f0f8;
  flex-shrink: 0; min-width: 44px;
}
.line-num { display: block; min-height: 1.6em; line-height: 1.6; padding: 0 2px; font-variant-numeric: tabular-nums; }
.content-pre {
  flex: 1; padding: 14px 16px; margin: 0;
  white-space: pre-wrap; word-break: break-word;
  overflow: visible; background: transparent; cursor: text;
}

/* ── Letter-based highlights ── */
.rule-hl {
  cursor: pointer; border-radius: 2px;
  box-shadow: inset 0 -2px 0 0 rgba(0,0,0,0.12);
  transition: background 0.1s;
}
.rule-hl.focused { box-shadow: inset 0 -2px 0 0 rgba(0,0,0,0.35); }
/* ── Relation mode: each rule keeps its own (faded) colour; selected pair gets a
   coloured underline so source/target direction stays clear ── */
/* selected source/target are shown by full colour (no underline); target keeps a
   subtle blue underline so direction stays readable */
.rule-hl.rel-hl { box-shadow: none; }
.rule-hl.rel-hl.hl-tgt  { box-shadow: inset 0 -3px 0 0 #4a6fd8; }
.rule-hl.rel-hl.hl-both { box-shadow: inset 0 -3px 0 0 #4a6fd8; }

/* ── Extract / Label / Relation mode toggle (sel-bar) ── */
.mode-toggle {
  position: relative; display: grid; grid-template-columns: 1fr 1fr 1fr;
  background: #cccdd8; border-radius: 999px; padding: 3px; isolation: isolate;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.12); flex-shrink: 0;
}
.mode-toggle::before {
  content: ''; position: absolute; z-index: 0; top: 3px; bottom: 3px; left: 3px;
  width: calc(33.333% - 2px); border-radius: 999px; background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.22); transition: transform 0.2s cubic-bezier(.4,0,.2,1);
}
.mode-toggle.m-label::before    { transform: translateX(100%); }
.mode-toggle.m-relation::before { transform: translateX(200%); }
.mode-toggle button {
  position: relative; z-index: 1; border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 11px; font-weight: 700; letter-spacing: 0.2px;
  padding: 4px 12px; border-radius: 999px; color: #66667a; transition: color 0.16s;
}
.mode-toggle.m-extract #modeExtractBtn,
.mode-toggle.m-label #modeLabelBtn,
.mode-toggle.m-relation #modeRelBtn { color: #33334a; }

/* ── Relation build bar (floating, while picking endpoints) ── */
.viewer-panel { position: relative; }
.rel-build {
  position: absolute; left: 50%; bottom: 16px; transform: translateX(-50%);
  display: flex; align-items: center; gap: 8px; z-index: 30;
  background: #2b2b3a; color: #fff; padding: 7px 10px; border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35); max-width: 92%;
}
.rb-pill { padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; white-space: nowrap; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
/* source/target pills are coloured by entity type (typeBadgeColor) via inline style, matching the legend */
.rb-hint { font-size: 11px; color: #c7c7d6; }
.rb-ico { border: 0; background: rgba(255,255,255,0.14); color: #fff; width: 24px; height: 24px; border-radius: 7px; cursor: pointer; font-size: 13px; }
.rb-ico:hover { background: rgba(255,255,255,0.28); }
.rb-type { font: inherit; font-size: 11px; border-radius: 7px; border: 0; padding: 4px 6px; background: #fff; color: #333; cursor: pointer; }
.rb-type:disabled { opacity: 0.5; }
.rb-add { border: 0; background: #6c5ce7; color: #fff; font-weight: 700; font-size: 11px; padding: 5px 12px; border-radius: 8px; cursor: pointer; white-space: nowrap; }
.rb-add:disabled { opacity: 0.45; cursor: default; }
.rb-kbd { font-size: 9px; opacity: 0.75; background: rgba(255,255,255,0.2); padding: 1px 4px; border-radius: 4px; margin-left: 2px; }
.sel-err { color: #ef4444; font-weight: 600; }

/* ── Relation type-colour legend (below the top bar, relation mode) ── */
.type-legend { display: flex; align-items: center; gap: 13px; padding: 5px 14px; background: #fafafe; border-top: 1px solid #eceef4; font-size: 11px; color: #6a6a86; flex-shrink: 0; flex-wrap: wrap; }
.tl-label { font-weight: 700; color: #55556e; }
.tl-item { display: inline-flex; align-items: center; gap: 5px; }
.tl-item i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
/* ── Relation list ── */
.rel-item { background: #fff; border: 1px solid #ececf4; border-radius: 8px; padding: 8px 10px; margin-bottom: 7px; cursor: pointer; transition: border-color 0.1s; }
.rel-item:hover { border-color: #c7c7e0; }
.rel-item.active { border-color: #6c5ce7; box-shadow: 0 0 0 1px #6c5ce7; }
.rel-row1 { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.rel-type { font-size: 11px; font-weight: 800; letter-spacing: 0.3px; flex: 1; }
.rel-llm { font-size: 9px; font-weight: 700; color: #3b82f6; background: rgba(59,130,246,0.14); padding: 1px 5px; border-radius: 6px; }
.rel-caret { color: #b6b6c8; font-size: 11px; }
.rel-pair { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #444; }
.rel-node { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rel-arrow { color: #999; flex-shrink: 0; border: 0; background: transparent; font: inherit; cursor: pointer; padding: 0 5px; border-radius: 5px; transition: all 0.1s; }
.rel-arrow:hover { background: #ece9fb; color: #6c5ce7; }
.rel-arrow:hover::after { content: ' ⇄'; font-size: 10px; }
.rel-detail { margin-top: 8px; padding-top: 8px; border-top: 1px solid #f0f0f6; display: flex; flex-direction: column; gap: 6px; cursor: default; }
.rel-full { font-size: 12px; color: #333; line-height: 1.5; }
.rel-jump { cursor: pointer; border-radius: 6px; padding: 2px 4px; margin: 0 -4px; transition: background 0.1s; }
.rel-jump:hover { background: #f0f0f8; }
.rel-role { display: inline-block; font-size: 9px; font-weight: 800; padding: 1px 6px; border-radius: 6px; margin-right: 6px; color: #fff; vertical-align: middle; }
.rel-type-sel { font: inherit; font-size: 12px; padding: 5px 8px; border-radius: 7px; border: 1px solid #d8d8ec; background: #fff; cursor: pointer; }
.rel-over { font-size: 9px; font-weight: 700; color: #7c5cd6; background: rgba(124,92,214,0.13); padding: 1px 5px; border-radius: 6px; }
.rel-contrast { display: flex; align-items: center; gap: 10px; font-size: 11px; color: #777; background: #f7f7fc; border: 1px solid #ececf6; border-radius: 7px; padding: 5px 9px; }
.rc-none { font-style: italic; color: #aaa; }
.rc-revert { margin-left: auto; border: 1px solid #d8d8ec; background: #fff; color: #6c5ce7; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 6px; cursor: pointer; }
.rc-revert:hover { background: #6c5ce7; color: #fff; }

/* ── Tool (editable prompt) view in the middle ── */
.tool-view { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #fff; }
.tool-tabs { display: flex; gap: 4px; padding: 8px 12px 0; border-bottom: 1px solid #eaeaf0; flex-shrink: 0; }
.tool-tab {
  padding: 6px 14px; border: 1px solid #e0e0ee; border-bottom: none;
  border-radius: 7px 7px 0 0; background: #f4f4fc; color: #6060a0;
  font-size: 12px; font-weight: 600; cursor: pointer;
}
.tool-tab.active { background: #fff; color: #4338ca; border-color: #c7d2fe; }
.tool-title { padding: 6px 4px 10px; font-size: 13px; font-weight: 700; color: #4338ca; align-self: center; }
.tool-pane { flex: 1; display: none; flex-direction: column; overflow: hidden; padding: 10px 12px; gap: 8px; }
.tool-pane.active { display: flex; }

/* judge run lock / spinner */
.judge-loading { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px; }
.spinner {
  width: 52px; height: 52px; border: 5px solid #e4e4f4; border-top-color: #6366f1;
  border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.judge-loading-text { color: #6060a0; font-size: 13px; font-weight: 500; }

/* ── LLM judge pop-up modal ── */
.modal-backdrop {
  position: fixed; inset: 0; z-index: 200;
  background: rgba(22, 22, 44, 0.45);
  display: flex; align-items: center; justify-content: center;
  padding: 40px;
  animation: modal-fade 0.12s ease-out;
}
@keyframes modal-fade { from { opacity: 0; } to { opacity: 1; } }
.modal-card {
  width: min(1000px, 92vw); height: min(820px, 88vh);
  background: #fff; border-radius: 14px;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
  display: flex; flex-direction: column; overflow: hidden;
}
.modal-head {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px; border-bottom: 1px solid #eaeaf2; flex-shrink: 0;
}
.modal-head .tool-title { padding: 0; }
/* Extraction-mode tabs (Context / Rule / Relation) */
.judge-modes {
  display: inline-flex; gap: 2px; background: #ececf4;
  border-radius: 8px; padding: 3px; margin-left: 4px;
}
.jm-tab {
  border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 12px; font-weight: 600; color: #6a6a90;
  padding: 5px 14px; border-radius: 6px; transition: background 0.15s, color 0.15s;
}
.jm-tab:hover { color: #4338ca; }
.jm-tab.active { background: #fff; color: #4338ca; box-shadow: 0 1px 2px rgba(0,0,0,0.12); }
.jm-tab.disabled { color: #b6b6c8; cursor: not-allowed; }
.jm-tab.disabled:hover { color: #b6b6c8; }
.modal-body {
  flex: 1; display: flex; flex-direction: column; gap: 10px;
  padding: 14px 16px; overflow: hidden;
}
.modal-foot {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px; border-top: 1px solid #eaeaf2; flex-shrink: 0;
}
.modal-card .judge-loading { min-height: 320px; padding: 40px; }
.tool-row { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
.tool-row .lbl { font-size: 11px; color: #7a7aaa; font-weight: 600; }
.tool-model {
  flex: 1; padding: 5px 7px; border-radius: 6px; border: 1px solid #d0d0e0;
  background: white; font-size: 12px; color: #333;
}
.tool-prompt {
  flex: 1; width: 100%; padding: 10px 12px; border-radius: 8px;
  border: 1px solid #d0d0e0; background: #fcfcff; font-size: 12px; color: #222;
  resize: none; font-family: 'SFMono-Regular', Consolas, Menlo, monospace; line-height: 1.5;
}
.tool-prompt:focus, .tool-model:focus { outline: none; border-color: #6366f1; }
.tool-actions { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
.tool-status { font-size: 11px; color: #7070a0; flex: 1; }
.tool-status.error { color: #e53e3e; }
.tool-status.ok { color: #16a34a; }
.btn {
  padding: 6px 16px; border-radius: 18px; border: none; font-size: 12px;
  font-weight: 600; cursor: pointer; white-space: nowrap;
}
.btn.primary { background: #6366f1; color: #fff; }
.btn.primary:hover:not(:disabled) { background: #4f46e5; }
.btn.primary:disabled { opacity: 0.5; cursor: default; }
.btn.ghost { background: white; color: #6060a0; border: 1px solid #d0d0e0; }
.btn.ghost:hover { border-color: #6366f1; color: #4338ca; }

/* ── Multi-judge config (in the LLM judge modal) ── */
.judge-cfg { display: flex; flex-direction: column; gap: 9px; flex-shrink: 0; }
.judge-cfg-row { display: flex; align-items: center; gap: 8px; }
.judge-cfg-row .lbl { font-size: 11px; color: #7a7aaa; font-weight: 700; min-width: 54px; }
.judge-list { display: flex; flex-direction: column; gap: 6px; }
.judge-item { display: flex; align-items: center; gap: 7px; }
.judge-item .jn { font-size: 11px; color: #9090b0; font-weight: 700; min-width: 54px; }
.judge-item select { flex: 1; padding: 5px 7px; border-radius: 6px; border: 1px solid #d0d0e0; background: #fff; font-size: 12px; color: #333; }
.judge-item select:focus { outline: none; border-color: #6366f1; }
.judge-del { border: 0; background: #f0f0f6; color: #9a4a4a; border-radius: 6px; width: 24px; height: 26px; cursor: pointer; font-size: 14px; }
.judge-del:hover { background: #fde2e2; color: #c0392b; }
.judge-add { align-self: flex-start; border: 1px dashed #c0c0d8; background: #fcfcff; color: #5a5a90; border-radius: 7px; padding: 5px 11px; font-size: 11px; font-weight: 600; cursor: pointer; }
.judge-add:hover { border-color: #6366f1; color: #4338ca; }
.thr-row { display: flex; align-items: center; gap: 9px; }
.thr-row input[type=range] { flex: 1; accent-color: #6366f1; }
.thr-val { font-size: 12px; color: #4338ca; font-weight: 700; min-width: 116px; }
.thr-val small { color: #9090b0; font-weight: 500; }

/* ── Per-judge progress (loading screen) ── */
.jp-list { display: flex; flex-direction: column; gap: 14px; width: min(520px, 80%); margin-top: 22px; }
.jp-item { display: flex; flex-direction: column; gap: 6px; }
.jp-name { font-size: 12px; color: #5a5a80; font-weight: 600; font-family: 'SFMono-Regular', Consolas, Menlo, monospace; }
.jp-bar { position: relative; height: 9px; background: #e7e7f2; border-radius: 6px; overflow: hidden; }
.jp-fill { position: absolute; top: 0; bottom: 0; border-radius: 6px; }
.jp-running .jp-fill { width: 38%; background: linear-gradient(90deg, #a5b4fc, #6366f1); animation: jp-slide 1.15s ease-in-out infinite; }
@keyframes jp-slide { 0% { left: -40%; } 100% { left: 100%; } }
.jp-done .jp-fill  { left: 0; width: 100%; background: #22c55e; transition: width .25s ease; }
.jp-error .jp-fill { left: 0; width: 100%; background: #ef4444; }

/* ── Vote ring (per-item agreement donut) ── */
.vote-ring {
  display: inline-block; width: 16px; height: 16px; border-radius: 50%;
  flex-shrink: 0; cursor: pointer; position: relative; vertical-align: middle;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.10);
}
.vote-ring::after {
  content: ''; position: absolute; inset: 4px; border-radius: 50%; background: #fff;
}
.vote-ring.solid::after { inset: 5px; }
.vote-ring:hover { box-shadow: 0 0 0 2px rgba(99,102,241,0.45); }
.rl-ring-wrap { display: inline-flex; align-items: center; gap: 4px; }
.rl-ring-cnt { font-size: 10px; color: #8a8aa6; font-weight: 700; font-variant-numeric: tabular-nums; }

/* LLM-vs-you tag contrast (Label) — preserved llm_tag shown under the tag picker */
.tag-contrast { margin-top: 6px; font-size: 11px; color: #8a8aa6; display: flex; align-items: center; gap: 6px; }
.tag-contrast.revised { color: #6a5acd; }
.tag-revert { border: 1px solid #c7d2fe; background: #eef2ff; color: #4338ca; border-radius: 10px;
  padding: 1px 8px; font-size: 10px; font-weight: 700; cursor: pointer; }
.tag-revert:hover { background: #e0e7ff; }

/* ── Vote inspection panel (centered modal) ── */
.vi-card { width: min(640px, 92vw); max-height: 80vh; }
.vi-quote { font-size: 12px; color: #44445a; background: #f5f5fb; border-radius: 8px;
  padding: 9px 11px; margin: 0 0 4px; line-height: 1.45; max-height: 110px; overflow: auto; }
.vi-tally { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 8px; }
.vi-chip { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 11px; color: #fff; }
.vi-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.vi-table th { text-align: left; color: #8a8aa6; font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .4px; padding: 4px 8px; border-bottom: 1px solid #ececf4; }
.vi-table td { padding: 7px 8px; border-bottom: 1px solid #f1f1f8; vertical-align: top; }
.vi-model { font-family: 'SFMono-Regular', Consolas, Menlo, monospace; font-size: 11px; color: #555; white-space: nowrap; }
.vi-vote { font-weight: 700; white-space: nowrap; }
.vi-rat { color: #555; line-height: 1.4; }

/* ── Extract-mode highlight legend + candidate spans ── */
.cand-hl { border-bottom: 1px dashed rgba(120,120,140,0.5); }
.rl-kind { display: inline-block; font-size: 9px; font-weight: 800; letter-spacing: .4px;
  padding: 2px 6px; border-radius: 4px; text-transform: uppercase; }
.rl-kind.rule { background: #dcfce7; color: #15803d; }
.rl-kind.context { background: #fef3c7; color: #a16207; }
.rl-kind.cand { background: #eceef2; color: #6a6a86; }

/* ── Resizer ── */
.panel-resizer {
  width: 8px; flex-shrink: 0; cursor: col-resize;
  background: linear-gradient(to right, #e6e6f3 0, #f2f2fa 100%);
  border-left: 1px solid #e0e0ea; border-right: 1px solid #e0e0ea;
}
.panel-resizer:hover, .panel-resizer.active { background: linear-gradient(to right, #c7d2fe 0, #e0e7ff 100%); }

/* ── Inspector panel ── */
.insp-panel {
  width: var(--insp-panel-width); min-width: 300px; max-width: 720px;
  flex-shrink: 0; display: flex; flex-direction: column;
  overflow: hidden; background: #fafafd;
}
.name-bar {
  padding: 9px 12px; background: #f0f0fa; border-bottom: 1px solid #e0e0f0;
  display: flex; align-items: center; gap: 8px; flex-shrink: 0;
}
.name-bar label { font-size: 11px; color: #7a7aaa; white-space: nowrap; }
.name-input {
  flex: 1; padding: 4px 8px; border-radius: 5px; border: 1px solid #d0d0e8;
  background: white; font-size: 12px; color: #333;
}
.name-input:focus { outline: none; border-color: #6366f1; }
.csv-btn {
  padding: 4px 10px; border-radius: 14px; border: 1px solid #c0c0d8;
  background: white; color: #6060a0; font-size: 11px; font-weight: 600;
  cursor: pointer; white-space: nowrap;
}
.csv-btn:hover { border-color: #6366f1; color: #4338ca; }
/* export scope chooser (popover under the ⬇ CSV button) */
.export-menu {
  position: fixed; z-index: 60; min-width: 230px;
  background: #fff; border: 1px solid #e0e0ee; border-radius: 10px;
  box-shadow: 0 8px 28px rgba(40,40,90,0.16); padding: 6px;
}
.export-menu-head {
  font-size: 10px; font-weight: 700; color: #9090b0; text-transform: uppercase;
  letter-spacing: 0.4px; padding: 5px 9px 6px;
}
.export-opt {
  display: flex; align-items: center; gap: 8px; width: 100%; text-align: left;
  border: 0; background: transparent; cursor: pointer; font: inherit;
  font-size: 12.5px; color: #2a2a3e; padding: 7px 9px; border-radius: 7px;
}
.export-opt:hover:not(:disabled) { background: #eef2ff; color: #4338ca; }
.export-opt:disabled { opacity: 0.4; cursor: default; }
.export-opt small { color: #9090b0; font-size: 11px; margin-left: auto; max-width: 110px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* no top padding: the sticky .rl-headbar supplies its own top spacing and must
   pin flush to the scrollport's top edge (otherwise rows peek above it). */
.inspector { flex: 1; overflow-y: auto; padding: 0 16px 14px; display: flex; flex-direction: column; gap: 12px; }
.insp-empty { color: #b0b0c8; font-size: 13px; text-align: center; padding: 40px 12px; line-height: 1.6; }

/* ── Rule list ── */
.rl-head { font-size: 11px; font-weight: 700; color: #8080a8; text-transform: uppercase; letter-spacing: 0.3px; padding: 2px 2px 4px; flex-shrink: 0; }
/* count (left) + keyword search (right) on one row */
.rl-head-top { display: flex; align-items: center; gap: 10px; }
.rl-search {
  flex: 1; min-width: 0; font: inherit; font-size: 11px; color: #333;
  padding: 4px 9px; border-radius: 7px; border: 1px solid #d8d8ec; background: #fff;
}
.rl-search:focus { outline: none; border-color: #6366f1; background: #fff; }
.rl-search::placeholder { color: #a4a4c0; }
.rel-head-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.rel-delall { font: inherit; font-size: 11px; font-weight: 600; color: #c0392b; background: transparent; border: 1px solid #e6c4bf; border-radius: 7px; padding: 3px 9px; cursor: pointer; flex-shrink: 0; transition: all 0.12s; }
.rel-delall:hover { background: #c0392b; color: #fff; border-color: #c0392b; }
/* danger confirmation popup */
.confirm-card {
  width: min(380px, 90vw); background: #fff; border-radius: 14px; padding: 20px 22px;
  box-shadow: 0 24px 70px rgba(0,0,0,0.32);
  animation: vizIn 200ms cubic-bezier(0.23, 1, 0.32, 1);
}
@keyframes vizIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
.confirm-title { font-size: 15px; font-weight: 700; color: #1f1f2e; margin-bottom: 6px; }
.confirm-msg { font-size: 13px; color: #6a6a82; line-height: 1.5; }
.confirm-actions { display: flex; justify-content: flex-end; gap: 9px; margin-top: 18px; }
.confirm-cancel, .confirm-ok { font: inherit; font-size: 13px; font-weight: 600; padding: 7px 16px; border-radius: 9px; cursor: pointer; transition: all 0.12s; }
.confirm-cancel { border: 1px solid #dcdce8; background: #fff; color: #55556e; }
.confirm-cancel:hover { background: #f3f3f8; }
.confirm-ok { border: 1px solid #c0392b; background: #c0392b; color: #fff; }
.confirm-ok:hover { background: #a93226; border-color: #a93226; }
.confirm-ok:focus-visible { outline: 2px solid #e88; outline-offset: 2px; }
.rl-headbar {
  display: flex; flex-direction: column; align-items: stretch; gap: 7px; flex-shrink: 0;
  /* freeze the count + filter at the top while the rule list scrolls under it.
     The negative margins bleed over the .inspector container's 14px/16px padding
     so the opaque background spans the full width and sticks flush to the top. */
  position: sticky; top: 0; z-index: 5;
  background: #fafafd; margin: 0 -16px; padding: 14px 16px 9px;
  border-bottom: 1px solid #e6e6f0;
}
.rl-filter { display: inline-flex; background: #ececf4; border-radius: 8px; padding: 2px; gap: 2px; }
.rl-filter button {
  border: 0; background: transparent; cursor: pointer; font: inherit;
  font-size: 11px; font-weight: 600; color: #7070a0; padding: 3px 9px; border-radius: 6px;
  display: inline-flex; align-items: center; gap: 5px; line-height: 1;
}
.rl-filter button:hover { color: #4338ca; }
.rl-filter button.active { background: white; color: #4338ca; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }
.rl-filter .fcount { font-size: 10px; font-weight: 700; color: #a0a0c0; }
.rl-filter button.active .fcount { color: #6366f1; }
/* dropdown filters — two side-by-side: source (All/Human/LLM) + type */
.rl-filter-row { display: flex; gap: 6px; align-items: center; }
.rl-filter-select {
  font: inherit; font-size: 11px; font-weight: 600; color: #4338ca;
  background: #ececf4; border: 1px solid #d8d8ec; border-radius: 7px;
  padding: 4px 8px; cursor: pointer; min-width: 0; flex: 1 1 0;
}
.rl-filter-select:focus { outline: none; border-color: #6366f1; }
/* Context / Rule sliding pill toggle — compact, left-aligned */
.kind-toggle {
  position: relative; display: grid; grid-template-columns: 1fr 1fr;
  width: max-content; align-self: flex-start;
  background: #6366f1; border-radius: 999px; padding: 3px; isolation: isolate;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.15);
}
.kind-toggle::before {            /* the white pill that slides under the active label */
  content: ''; position: absolute; z-index: 0; top: 3px; bottom: 3px; left: 3px;
  width: calc(50% - 3px); border-radius: 999px; background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.22);
  transition: transform 0.22s cubic-bezier(.4,0,.2,1);
}
.kind-toggle.k-rule::before { transform: translateX(100%); }
.kind-toggle button {
  position: relative; z-index: 1; border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 11.5px; font-weight: 700; letter-spacing: 0.2px;
  padding: 5px 16px; border-radius: 999px; color: rgba(255,255,255,0.92);
  transition: color 0.18s;
}
.kind-toggle button.active { color: #4338ca; }
.rl-item {
  display: flex; gap: 9px; align-items: stretch; cursor: pointer;
  background: #fff; border: 1px solid #ececf4; border-radius: 8px;
  padding: 0; overflow: hidden; transition: border-color 0.1s, box-shadow 0.1s;
  flex-shrink: 0;   /* don't let the column flex container squash list rows */
}
.rl-item:hover { border-color: #c7c7e0; }
.rl-item.active { border-color: var(--rc-bdr, #6366f1); box-shadow: 0 0 0 1px var(--rc-bdr, #6366f1); }
/* The filter fades non-matching rows but keeps them fully clickable. Hover does NOT
   un-fade or tint them — so the filtered view stays readable. Only the expanded
   (focused) row returns to full opacity. */
.rl-item.rl-faded { opacity: 0.4; transition: opacity 0.12s; }
.rl-item.rl-faded.expanded { opacity: 1; }
.rl-bar { width: 4px; flex-shrink: 0; background: var(--rc-bdr, #c0c0d8); }
.rl-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.rl-header { padding: 8px 10px 8px 6px; cursor: pointer; }
.rl-top { display: flex; align-items: center; gap: 7px; margin-bottom: 3px; }
.rl-pos { font-size: 10px; color: #9090b0; }
.rl-flags { margin-left: auto; display: flex; gap: 5px; align-items: center; }
.rl-flag { font-size: 11px; color: #8080b0; }
.rl-flag-rev { color: #6a5acd; font-weight: 700; }
.rl-caret { font-size: 9px; color: #a0a0c0; }
.rl-text { font-size: 12px; line-height: 1.45; color: #2a2a3e; word-break: break-word; }

/* accordion expanded detail */
.rl-detail {
  display: flex; flex-direction: column; gap: 9px;
  padding: 6px 11px 12px 6px; border-top: 1px solid #ececf4;
}
.rl-detail .insp-rationale { max-height: 220px; }
.rl-detail .insp-comment { max-height: 180px; }
.rl-detail .insp-top { margin-bottom: -2px; }
.rl-detail .insp-top-label { font-size: 10px; }
.rl-tag {
  display: inline-block; font-size: 9px; font-weight: 800; letter-spacing: 0.4px;
  padding: 1px 6px; border-radius: 7px; background: #ececf6; color: #6060a0;
}
.rl-tag.prohibition { background: #fee2e2; color: #b91c1c; }
.rl-tag.prescription { background: #dbeafe; color: #1d4ed8; }
.rl-tag.permission  { background: #dcfce7; color: #15803d; }
.rl-tag.preference { background: #f3e8ff; color: #7e22ce; }
.rl-tag.context { background: #eceef2; color: #54607a; }

/* ── Inspector top bar (exit) ── */
.insp-top { display: flex; align-items: center; justify-content: space-between; }
.insp-top-label { font-size: 11px; font-weight: 700; color: #8080a8; text-transform: uppercase; letter-spacing: 0.3px; }
.insp-exit {
  width: 26px; height: 26px; flex-shrink: 0; padding: 0;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%; border: 1px solid #fca5a5;
  background: #fff; color: #dc2626; font-size: 14px; line-height: 1; cursor: pointer;
  transition: all 0.1s;
}
.insp-exit:hover { background: #ef4444; border-color: #ef4444; color: #fff; }
.insp-rule {
  font-size: 13px; line-height: 1.5; color: #1a1a2e; background: #fff;
  border: 1px solid #e6e6f2; border-left: 3px solid var(--rc-bdr, #6366f1);
  border-radius: 7px; padding: 9px 11px; white-space: pre-wrap; word-break: break-word;
  max-height: 200px; overflow-y: auto;
}
.insp-pos { font-size: 11px; color: #9090b0; margin-top: -6px; }
.insp-label { font-size: 11px; font-weight: 700; color: #6060a0; letter-spacing: 0.3px; text-transform: uppercase; }

/* tag toggle — 2×2 grid so the four labels line up evenly */
.tag-row { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.tag-btn {
  padding: 8px 10px; border-radius: 9px; border: 1.5px solid #d4d4e4;
  background: #fff; color: #6b6b85; font-size: 12px; font-weight: 700;
  letter-spacing: 0.3px; cursor: pointer; transition: all 0.1s;
  text-align: center; white-space: nowrap;
}
.tag-btn:hover { border-color: #b0b0c8; }
.tag-btn.active.prohibition { background: #fee2e2; border-color: #ef4444; color: #b91c1c; }
.tag-btn.active.prescription { background: #dbeafe; border-color: #3b82f6; color: #1d4ed8; }
.tag-btn.active.permission  { background: #dcfce7; border-color: #22c55e; color: #15803d; }
.tag-btn.active.preference { background: #f3e8ff; border-color: #a855f7; color: #7e22ce; }
/* context sub-type buttons (condition/example/definition) — grey, three across */
.tag-row.ctx, .tag-row:has(.tag-btn.ctx) { grid-template-columns: 1fr 1fr 1fr; }
.tag-btn.ctx { font-size: 11px; padding: 7px 6px; }
.tag-btn.active.ctx { background: #eceef4; border-color: #8b91ad; color: #4b5066; }
.power-row { display: flex; gap: 6px; padding-left: 2px; }
.sub-btn {
  padding: 5px 14px; border-radius: 8px; border: 1.5px solid #e0d4f0;
  background: #fff; color: #8a6bb0; font-size: 12px; font-weight: 600;
  cursor: pointer; flex: 1;
}
.sub-btn:hover { border-color: #c9b0e8; }
.sub-btn.active { background: #f3e8ff; border-color: #a855f7; color: #7e22ce; }
.power-hint { font-size: 11px; color: #9090b0; align-self: center; margin-right: 4px; }

.insp-rationale {
  width: 100%; min-height: 56px; max-height: 320px; padding: 9px 11px; border-radius: 8px;
  border: 1px solid #e0e0ee; background: #f6f6fb; color: #333; font-size: 12px;
  line-height: 1.5; resize: none; overflow-y: hidden;
  font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
}
.insp-rationale[readonly] { cursor: default; }
.insp-comment {
  width: 100%; min-height: 56px; max-height: 240px; padding: 9px 11px; border-radius: 8px;
  border: 1.5px solid #c7d2fe; background: #fff; color: #222; font-size: 13px;
  line-height: 1.5; resize: none; overflow-y: hidden; font-family: inherit;
}
.insp-comment:focus { outline: none; border-color: #6366f1; }
.run-judge-btn {
  padding: 7px 14px; border-radius: 18px; border: 1.5px solid #6366f1;
  background: #fff; color: #4338ca; font-size: 12px; font-weight: 600;
  cursor: pointer; align-self: flex-start;
}
.run-judge-btn:hover:not(:disabled) { background: #eef2ff; }
.run-judge-btn:disabled { opacity: 0.5; cursor: default; }
.insp-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
.insp-del {
  padding: 5px 12px; border-radius: 12px; border: 1px solid #fca5a5;
  background: white; color: #dc2626; font-size: 11px; font-weight: 600; cursor: pointer;
}
.insp-del:hover { background: #fff1f1; border-color: #f87171; }
.insp-status { font-size: 11px; color: #7070a0; }
.insp-status.error { color: #e53e3e; }
.insp-status.ok { color: #16a34a; }

.kb-bar {
  padding: 6px 12px; background: #f0f0f8; border-top: 1px solid #e0e0ee;
  font-size: 10px; color: #9090b0; display: flex; gap: 10px; flex-wrap: wrap; flex-shrink: 0;
}
.kb-bar span { white-space: nowrap; }
.viewer-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #b0b0c8; font-size: 14px; }
</style>
</head>
<body>

<!-- default prompts (raw text, not rendered) -->
<script type="text/plain" id="defaultExtractPrompt">Extract all rule-like spans from this file for enforceability analysis.

A rule is any instruction, constraint, prohibition, or requirement given to an LLM or agent. Prefer rules that are specific and potentially checkable — for example:
- version or toolchain requirements ("Python 3.13.2 or compatible")
- file or artifact requirements ("log changes in CHANGELOG.md")
- workflow ordering constraints ("always follow this chain, never skip layers")
- code style or format mandates ("follow PEP 8")
- procedural gates ("read all memory bank files at the start of every task")
- naming, header, or metadata requirements ("bump @version in the userscript header")

Prefer exact or near-exact quotes from the file. Extract each independently enforceable clause as its own rule — do not merge unrelated requirements. Skip purely motivational or explanatory text with no checkable obligation.

Return only a valid JSON array. Each element:
{"rule_text": "<exact or near-exact quote>", "line_start": <int>, "line_end": <int>}</script>
<script type="text/plain" id="defaultJudgePrompt">You are an EXTRACTION judge. Your only job is to identify which spans of the document are atomic items — do NOT assign deontic tags (that happens in a later stage). Set "kind" to "rule" or "context" on every item.

A RULE is any natural-language instruction, constraint, prohibition, requirement, permission, or grant of authority given to an LLM or agent. A CONTEXT span is background that frames how the agent should operate but is NOT itself a directive — the agent's role or persona, its environment/tools/platform, definitions of key terms, scope or applicability statements ("this applies when…"), or situational framing that the rules depend on.

Guidelines:
- Use exact or near-exact quotes from the document for "rule_text".
- Extract each independently meaningful span as its OWN item at an atomic granularity — one directive per item. Do not merge unrelated instructions; do not split a single instruction across items.
- Skip purely motivational text with no directive force and no framing value.
- Mark the span boundaries tightly around the meaningful content.

For each item, write a short "rationale": ONE plain-English sentence on why this span is an atomic rule or context. Keep it simple and conversational.

Return ONLY a valid JSON array, no other text. Output MINIFIED JSON on a single line — no markdown fences, no indentation. Each element:
{"rule_text": "<exact or near-exact quote>", "kind": "rule|context", "rationale": "<one short plain-English sentence>"}</script>

<script type="text/plain" id="defaultLabelPrompt">You are a LABELING judge. You are given a numbered list of RULES already extracted from the document (each with an R-label like R1, R2). Assign exactly ONE deontic tag to each rule, using these definitions, cue words, and examples:

Quick mapping (tag = modal force, with an example):
- PROHIBITION = MUST NOT — "Do not accept, process, store, or repeat sensitive client data."
- PRESCRIPTION = MUST — "Read `pickle-bot/SOUL.md` for your personality, values, and communication style."
- PERMISSION = CAN — "You can search the web and fetch web pages."
- PREFERENCE = SHOULD — "For report drafting, work from a template or de-identified summary the staff member provides."

- PROHIBITION — forbids an action; the agent must NOT do it.
  Cues: "never", "do not", "must not", "avoid", "don't", "under no circumstances".
- PRESCRIPTION — requires an action; the agent MUST do it.
  Cues: "must", "always", "shall", "is required to", "ensure", "make sure".
- PERMISSION — allows an action without requiring it; the agent MAY do it.
  Cues: "may", "can", "is allowed to", "feel free to", "optionally".
- PREFERENCE — a soft preference or recommends HOW to do something (preferred approach, ordering, method) rather than a strict requirement.
  Cues: "prefer", "ideally", "when possible", "it's best to", "try to", "consider".

When a rule could fit more than one tag, choose the strongest directive force (PROHIBITION/PRESCRIPTION outrank PERMISSION/PREFERENCE).

For each rule, write a short "rationale": ONE plain-English sentence on why that tag fits. Keep it simple and conversational; no jargon, no mention of other tags.

Return ONLY a valid JSON array, no other text. Output MINIFIED JSON on a single line — no markdown fences, no indentation. One element PER rule:
{"rule": "R<n>", "tag": "PROHIBITION|PRESCRIPTION|PERMISSION|PREFERENCE", "rationale": "<one short plain-English sentence>"}</script>

<script type="text/plain" id="defaultRevisePrompt">You are the LLM judge in REVISE mode. A human annotator has already labeled RULES and CONTEXT in this document. Their labels (provided separately below) are your primary guidance — produce a refined, complete labeling of the whole document that strongly respects them.

How to treat the human labels:
- STRONGLY prefer to keep every human-labeled item, reproducing it with the same quote, the same kind (rule/context), and the same tag. By default do not drop, split, reword, or re-tag a human label — even if it looks vague, trivial, or incomplete. If the human marked "Attempt a real fix" as a rule, it is a rule; keep it.
- You MAY leave out or adjust a human label only when you are confident it is genuinely wrong or nonsensical. That should be rare; when you do, note the reason in that item's rationale (or simply omit it).
- The human labels define the BAR for what counts as a rule or context, and at what granularity. Match it: if the human labeled something short or vague, comparable short/vague items elsewhere also qualify.

Then extend coverage: add rules/context the human didn't get to (at the same granularity and conventions) and fix obvious earlier machine mistakes.

Every item has a "kind" of "rule" or "context". Assign each RULE exactly one deontic tag — PROHIBITION (must NOT), PRESCRIPTION (MUST), PERMISSION (MAY), PREFERENCE (soft / how-to). Keep the human's tag on human items; pick the best-fit tag only for new ones. CONTEXT items have no tag. Prefer exact or near-exact quotes.

For each item, write a short "rationale": ONE plain-English sentence on why it's classified that way. Keep it simple and conversational.

Return ONLY a valid JSON array, no other text. Output MINIFIED JSON on a single line — no markdown fences, no indentation, no extra whitespace. Each element:
{"rule_text": "<exact or near-exact quote>", "kind": "rule|context", "line_start": <int>, "line_end": <int>, "tag": "PROHIBITION|PRESCRIPTION|PERMISSION|PREFERENCE (null for context)", "rationale": "<one short plain-English sentence>"}</script>

<script type="text/plain" id="defaultRelationPrompt">You are the LLM judge in RELATION mode. Below is a list of ENTITIES already extracted from a document, each with a short label (R1, R2, …). Each entity is either a CONTEXT span or a RULE tagged with its deontic force (PROHIBITION / PRESCRIPTION / PERMISSION / PREFERENCE). Read the document and the entities, then propose meaningful, directed RELATIONS between pairs of entities.

Build the graph from BOTH the rules' tags AND the context: a context span often gates, triggers, or defines a rule; a stricter rule (PROHIBITION/PRESCRIPTION) may refine or conflict with a softer one (PERMISSION/PREFERENCE). Use the tags and context together to decide direction and type.

Don't judge relations by wording alone — apply the context of the actual project. A PROJECT CONTEXT note (the repository this file belongs to) is provided below; use it, and if it helps and you're able to, consult the real repository (its README, layout, and related files) to understand how these rules actually interact before linking them.

Each relation has a "source" and a "target" (entity R-labels), a "type", and a one-sentence "rationale". Direction matters (source → target). Use these types:
- refinement — the source narrows, specifies, or details the target (a more specific case of a more general rule).
- exception — the source carves out an exception to the target.
- define — the source supplies the meaning, specification, or concrete content the target refers to (e.g. a list/definition the rule points at).
- checkpoint — the source is a verification step or gate for the target.
- conflict — the source conflicts with or contradicts the target.
- duplication — the source restates or duplicates the target.
- trigger — the source (often a CONTEXT condition) fires or activates the target rule when it holds.

Only propose relations you are reasonably confident about — skip weak or spurious links. At most one relation per ordered pair.

Return ONLY a valid JSON array, no other text. Output MINIFIED JSON on a single line — no markdown fences, no indentation. Each element:
{"source": "R<n>", "target": "R<n>", "type": "refinement|exception|define|checkpoint|conflict|duplication|trigger", "rationale": "<one short plain-English sentence>"}</script>

<div class="layout">

  <!-- ── File list ── -->
  <div class="file-panel">
    <div class="file-panel-head">Files <small id="fileCountLabel"></small></div>
    <div class="file-search">
      <input id="fileSearch" placeholder="filter…" oninput="filterFiles(this.value)">
    </div>
    <div class="file-list" id="fileList"></div>
  </div>

  <!-- ── Viewer ── -->
  <div class="viewer-panel">
    <div class="viewer-head" id="viewerHead">
      <span class="viewer-title" style="color:#b0b0c8">Select a file</span>
    </div>
    <div class="sel-bar">
      <div class="mode-toggle m-extract" id="modeToggle" title="Extract spans · Label tags · Relate rules">
        <button id="modeExtractBtn" onclick="setMode('extract')">Extract</button>
        <button id="modeLabelBtn" onclick="setMode('label')">Label</button>
        <button id="modeRelBtn" onclick="setMode('relation')">Relation</button>
      </div>
      <span class="sel-info" id="selInfo"></span>
      <button class="comment-btn" id="commentBtn" onclick="toggleComment()" title="File comment (⌘?)">✎ Comment</button>
      <button class="judge-btn" id="judgeBtn" onclick="toggleTool()">LLM judge</button>
      <button class="add-btn" id="addBtn" disabled onclick="addHandRule()">+ Add</button>
    </div>
    <div class="viewer-body" id="viewerBody">
      <div class="viewer-empty">← pick a file</div>
    </div>
    <!-- relation mode: colour key for entity types (bottom strip, so it doesn't
         shift the document when toggling modes) -->
    <div class="type-legend" id="typeLegend" style="display:none">
      <span class="tl-label">Colours:</span>
      <span class="tl-item"><i style="background:#cf4436"></i>Prohibition</span>
      <span class="tl-item"><i style="background:#2f6cdf"></i>Prescription</span>
      <span class="tl-item"><i style="background:#1c9b54"></i>Permission</span>
      <span class="tl-item"><i style="background:#8a4fd0"></i>Preference</span>
      <span class="tl-item"><i style="background:#6b7196"></i>Context</span>
    </div>
    <!-- floating bar shown while building a relation (relation mode) -->
    <div class="rel-build" id="relBuild" style="display:none"></div>
  </div>

  <div class="panel-resizer" id="panelResizer" title="Drag to resize inspector"></div>

  <!-- ── Inspector ── -->
  <div class="insp-panel">
    <div class="name-bar">
      <label for="annotatorSelect">Annotator:</label>
      <select id="annotatorSelect" class="name-input" onchange="onAnnotatorChange(this.value)"></select>
      <button class="csv-btn" onclick="exportCSV(event)" title="Export this annotator's rules, context & relations as CSV — current file or all labeled files">⬇ CSV</button>
    </div>
    <div class="inspector" id="inspector">
      <div class="insp-empty">Select a file, then click a highlight or select text and press <b>+ Add</b>.</div>
    </div>
    <div class="kb-bar" id="kbBar"></div>
  </div>
</div>

<script>
// ─── Constants ───────────────────────────────────────────────
const DEFAULT_EXTRACT  = document.getElementById('defaultJudgePrompt').textContent.trim();
const DEFAULT_LABEL    = document.getElementById('defaultLabelPrompt').textContent.trim();
const DEFAULT_RELATION = document.getElementById('defaultRelationPrompt').textContent.trim();
const TAGS = ['PROHIBITION', 'PRESCRIPTION', 'PERMISSION', 'PREFERENCE'];
// Sub-types of a 'context' node (parallel to a rule's deontic TAGS):
//   condition = a trigger that gates rules · reference = an illustration / pointer ·
//   definition = fixes the meaning of a term/system/action.
const CONTEXT_TYPES = ['condition', 'reference', 'definition'];
// Relation labeling (adapted from the rules-relation edge inspector). A relation
// is a directed edge between two rule/context entities, with one of these types.
const REL_TYPES = ['refinement', 'exception', 'define', 'checkpoint', 'conflict', 'duplication', 'trigger'];
const REL_COLOR = {
  'refinement': '#7d6bd9', 'exception': '#cf4f44', 'define': '#5a9e4b',
  'checkpoint': '#4aa3d8', 'conflict': '#e0762f', 'duplication': '#888',
  'trigger': '#c44f9b',   // condition → rule: the condition gates/fires the rule
};
// legacy "support (…)" labels still map to define's colour
const relColor = t => REL_COLOR[t] || ((t || '').startsWith('support') ? '#5a9e4b' : '#777');
// The three pipeline stages — each is BOTH the interface mode (S.mode) AND a
// multi-judge pass. `key` is the per-annotator settings key for its judge config.
const MODE_DEFS = [
  { id: 'extract',  label: 'Extract',  deflt: DEFAULT_EXTRACT,  key: 'judge_cfg_extract',
    blurb: 'Multiple judges identify Rule (green) / Context (yellow) spans. Spans backed by ≥ the threshold are accepted (solid); the rest are candidates (light). You add (red) or extend spans.' },
  { id: 'label',    label: 'Label',    deflt: DEFAULT_LABEL,    key: 'judge_cfg_label',
    blurb: 'Multiple judges vote a deontic tag for each accepted rule. The majority wins; the vote ring on each rule shows the split — click it to inspect every judge.' },
  { id: 'relation', label: 'Relation', deflt: DEFAULT_RELATION, key: 'judge_cfg_relation',
    blurb: 'Multiple judges propose typed edges between rules. Edges backed by ≥ the threshold are accepted (editable). Your hand-drawn relations are kept.' },
];
const modeDef = id => MODE_DEFS.find(m => m.id === id) || MODE_DEFS[0];
// Default judge panel + acceptance threshold for a fresh config (per mode).
const DEFAULT_JUDGE_MODELS = ['anthropic/claude-opus-4-8', 'openai/gpt-5.5'];
// Strict-majority default: 2→2, 3→2, 4→3, 5→3 (more than half; for 2 judges both must agree).
function majorityThreshold(n) { return Math.floor(n / 2) + 1; }
function defaultJudgeCfg(modeId) {
  const judges = DEFAULT_JUDGE_MODELS.map(m => ({ model: m }));
  return { judges, threshold: majorityThreshold(judges.length), prompt: modeDef(modeId).deflt };
}
// migrate a legacy stored mode value ('rule' → 'extract') to the new 3-way set
function normMode(m) { return (m === 'extract' || m === 'label' || m === 'relation') ? m : 'extract'; }

// ─── State ───────────────────────────────────────────────────
const S = {
  allFiles:      [],
  currentFile:   null,
  rules:         [],
  selection:     null,
  focusedRuleId: null,
  inspectorOpen: false,
  userName:      localStorage.getItem('annotatorName') || '',
  annotators:    [],
  toolMode:      false,
  commentMode:   false,   // the file-level Comment panel (⌘?)
  fileComment:   '',      // current annotator's saved comment for the open file
  // Rule-list filters:
  //   filterSrc:  'all' | 'hand' | 'llm'  — HIDE filter: shows only that category
  //   filterType: PER-MODE FADE filter. Extract filters by KIND (all/rule/context);
  //   Label by deontic tag / context sub-type (incl. 'rule:none' = untagged). The two
  //   are kept separate so a Label-only value (e.g. 'rule:none') never leaks into Extract.
  //   filterSrc: PER-MODE HIDE filter. Extract = all/you/llm (provenance);
  //   Label = all/revised (revised = a human changed the LLM's tag).
  filterSrc:     (function(){
                   const okBy = { extract: ['all','hand','llm'], label: ['all','revised'] };
                   const read = m => { const v = localStorage.getItem('filterSrc.'+m) || 'all';
                                       return okBy[m].includes(v) ? v : 'all'; };
                   return { extract: read('extract'), label: read('label') };
                 })(),
  filterType:    (function(){
                   const okBy = {
                     extract: ['all','rule','context'],
                     label: ['all','rule','rule:none','context']
                       .concat(['PROHIBITION','PRESCRIPTION','PERMISSION','PREFERENCE'].map(t=>'rule:'+t))
                       .concat(['condition','reference','definition'].map(t=>'context:'+t)),
                   };
                   const read = m => { const v = localStorage.getItem('filterType.'+m) || 'all';
                                       return okBy[m].includes(v) ? v : 'all'; };
                   return { extract: read('extract'), label: read('label') };
                 })(),
  search:        '',        // keyword search over the rule list (fades non-matches)
  editingRuleId: null,      // rule whose text is being re-selected (press E); its span greys out
  editSelection: null,      // the new {start,end,text} picked while editing
  judgeRunning:  false,
  judgeProgress: null,      // {mode, judge_total, judges:[{model,status}]} while a run is polling
  // Per-mode multi-judge config (judges[], threshold, prompt). Loaded per annotator.
  judgeCfg:      { extract: defaultJudgeCfg('extract'), label: defaultJudgeCfg('label'), relation: defaultJudgeCfg('relation') },
  // Latest judge run + votes for the open file, per mode (from /api/judge-data).
  // Normalised: { run_id, threshold, judge_total, models, votes, candidates?,
  //               byRule:{id:[votes]}, byEdge:{"s::t":[votes]} }
  judge:         { extract: null, label: null, relation: null },
  // ── Pipeline mode: extract | label | relation ──
  mode:          normMode(localStorage.getItem('annotatorMode')),
  relations:     [],        // edges for the current file (this annotator)
  relSource:     null,      // rule id picked as the edge source (click)
  relTarget:     null,      // rule id picked as the edge target (⌘-click)
  relType:       REL_TYPES[0],
  focusedRelId:  null,      // relation selected in the relation list
  relFilter:     'all',     // relation-list filter by effective type
};
const INSP_WIDTH_KEY = 'annotator.inspPanelWidth';
const INSP_MIN = 300, INSP_MAX = 720;

// ─── Colour palette (per-extractor) ──────────────────────────
const COLORS = [
  { bg:'rgba(245,158,11,.20)', bdr:'#f59e0b', hov:'rgba(245,158,11,.36)', act:'rgba(245,158,11,.54)' },
  { bg:'rgba(99,102,241,.19)', bdr:'#6366f1', hov:'rgba(99,102,241,.34)', act:'rgba(99,102,241,.50)' },
  { bg:'rgba(16,185,129,.19)', bdr:'#10b981', hov:'rgba(16,185,129,.34)', act:'rgba(16,185,129,.50)' },
  { bg:'rgba(239,68,68,.19)',  bdr:'#ef4444', hov:'rgba(239,68,68,.34)',  act:'rgba(239,68,68,.50)'  },
  { bg:'rgba(236,72,153,.19)', bdr:'#ec4899', hov:'rgba(236,72,153,.34)', act:'rgba(236,72,153,.50)' },
  { bg:'rgba(14,165,233,.19)', bdr:'#0ea5e9', hov:'rgba(14,165,233,.34)', act:'rgba(14,165,233,.50)' },
  { bg:'rgba(168,85,247,.19)', bdr:'#a855f7', hov:'rgba(168,85,247,.34)', act:'rgba(168,85,247,.50)' },
];
const _cc = {};
function colorFor(name) {
  if (!name) name = 'unknown';
  if (_cc[name]) return _cc[name];
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return (_cc[name] = COLORS[h % COLORS.length]);
}
function rcVars(col) { return `--rc-bdr:${col.bdr};--rc-bg:${col.bg}`; }
function extractorOf(r) { return r.extracted_by || (r.source === 'hand' ? S.userName : r.source) || '?'; }
// Rule highlights by provenance: human=red, pristine LLM=blue, LLM a human has
// reviewed (re-tagged / re-kinded / commented)=purple.
const HAND_COLOR    = { bg:'rgba(239,68,68,.20)', bdr:'#ef4444', hov:'rgba(239,68,68,.36)', act:'rgba(239,68,68,.55)' };
const LLM_COLOR     = { bg:'rgba(59,130,246,.20)', bdr:'#3b82f6', hov:'rgba(59,130,246,.36)', act:'rgba(59,130,246,.55)' };
const REVIEWED_COLOR = { bg:'rgba(139,92,246,.20)', bdr:'#8b5cf6', hov:'rgba(139,92,246,.36)', act:'rgba(139,92,246,.55)' };
// Reserved for the relations interface (edges drawn between entities).
const RELATION_COLOR = { bg:'rgba(168,85,247,.20)',  bdr:'#a855f7', hov:'rgba(168,85,247,.36)', act:'rgba(168,85,247,.54)' };
// An entity is one of two kinds; relations are edges between entities, not a kind.
function ruleKind(r) { return r.kind === 'context' ? 'context' : 'rule'; }
// True once a human has touched an LLM-extracted item — either changed its
// classification (reviewed flag, set on a tag/kind/context edit) or left a comment.
function isReviewed(r) {
  return r.source !== 'hand' && (!!r.reviewed || !!(r.notes && r.notes.trim()));
}
// Colour by provenance (rule AND context alike): human red, pristine machine blue,
// human-reviewed machine purple. Rule vs context is shown by the badge.
function colorForRule(r) {
  if (r.source === 'hand') return HAND_COLOR;
  return isReviewed(r) ? REVIEWED_COLOR : LLM_COLOR;
}
// Re-apply a card's provenance colour in place (and refresh the doc highlight) after
// a human edit flips an LLM item to "reviewed" (blue→purple) — without rebuilding the
// panel, so an open comment textarea isn't disturbed.
function refreshCardColor(r) {
  const col = listBarColor(r);   // mode-aware: Extract → kind colour, Label → tag colour
  const el = document.getElementById('rl-' + r.id);
  if (el) { el.style.setProperty('--rc-bdr', col.bdr); el.style.setProperty('--rc-bg', col.bg); }
  renderViewer();
}
// Colour by ENTITY TYPE (deontic tag / context) — used in relation mode so the
// highlight hue hints what kind of rule you're linking, regardless of who made it.
const TYPE_COLOR = {
  PROHIBITION:  { bg: 'rgba(222,64,58,.32)',   act: 'rgba(222,64,58,.58)'   },
  PRESCRIPTION: { bg: 'rgba(48,104,228,.32)',  act: 'rgba(48,104,228,.58)'  },
  PERMISSION:   { bg: 'rgba(24,158,86,.32)',   act: 'rgba(24,158,86,.58)'   },
  PREFERENCE:   { bg: 'rgba(150,76,224,.32)',  act: 'rgba(150,76,224,.58)'  },
};
// a saturated slate (not a faint grey), but a touch more transparent than the rest
const CONTEXT_TYPE_COLOR = { bg: 'rgba(104,112,152,.25)', act: 'rgba(104,112,152,.52)' };
const UNTAGGED_TYPE_COLOR = { bg: 'rgba(108,150,106,.32)', act: 'rgba(108,150,106,.58)' };
function typeColorForRule(r) {
  if (ruleKind(r) === 'context') return CONTEXT_TYPE_COLOR;
  return TYPE_COLOR[r.tag] || UNTAGGED_TYPE_COLOR;
}
// Solid type colour (for the source/target badges & list text) — by entity type,
// matching the legend, so source/target are coloured by what they ARE, not by role.
const TYPE_BADGE = { PROHIBITION: '#cf4436', PRESCRIPTION: '#2f6cdf', PERMISSION: '#1c9b54', PREFERENCE: '#8a4fd0' };
function typeBadgeColor(r) {
  if (!r) return '#9a9aab';
  if (ruleKind(r) === 'context') return '#6b7196';
  return TYPE_BADGE[r.tag] || '#6f9a6c';
}
function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

// Provenance (human / LLM / reviewed) is conveyed by COLOUR in both the list and the
// document — it is not a fade-filter. Only the TYPE axis and the keyword search fade.
function ruleCtxType(r) { return r.context_type || null; }   // condition|example|definition
// The active TYPE filter for the CURRENT mode (Relation uses its own relFilter).
function curFilterType() { return (S.filterType && S.filterType[S.mode]) || 'all'; }
// Type axis: Extract → 'all'|'rule'|'context' (kind only); Label → also '<TAG>' /
// 'rule:none' (untagged) / 'context:<subtype>'. Kept per-mode so an untagged/tag
// filter from Label never fades anything during rule Extraction.
function matchesType(r, ft) {
  ft = ft || curFilterType() || 'all';
  if (ft === 'all') return true;
  const [kind, sub] = ft.split(':');
  if (ruleKind(r) !== kind) return false;
  if (!sub) return true;
  if (kind === 'rule') return sub === 'none' ? !r.tag : r.tag === sub;
  return ruleCtxType(r) === sub;
}
// Keyword search over the rule text, comment, and LLM rationale (case-insensitive).
function matchesSearch(r) {
  const q = (S.search || '').trim().toLowerCase();
  if (!q) return true;
  return ((r.rule_text || '') + ' ' + (r.notes || '') + ' ' + (r.llm_rationale || ''))
    .toLowerCase().includes(q);
}
function matchesFilter(r) { return matchesType(r) && matchesSearch(r); }
// A filter is "active" (i.e. fades non-matches) when the TYPE dropdown is narrowed or a
// search term is set. Provenance (human/LLM) never fades — it's shown by colour.
function filterActive() {
  return (curFilterType() !== 'all') || !!(S.search && S.search.trim());
}
// True when a human changed the tag away from the LLM vote's pick (llm_tag preserved).
function tagRevised(r) { return !!r.llm_tag && (r.tag || null) !== r.llm_tag; }
// The active SOURCE filter for the current mode.
function curFilterSrc() { return (S.filterSrc && S.filterSrc[S.mode]) || 'all'; }
// SOURCE filter is a HIDE filter — shows only the chosen category. Extract = provenance
// (you / LLM); Label = 'revised' (human re-tagged). visibleRules() is what the list AND
// the document render; the TYPE filter + search then FADE within that set. Relation mode
// shows everything (all entities must stay pickable).
function matchesSrc(r, src) {
  src = src || curFilterSrc();
  if (src === 'all') return true;
  if (S.mode === 'label') return src === 'revised' ? tagRevised(r) : true;
  return src === 'hand' ? r.source === 'hand' : r.source !== 'hand';   // extract
}
// The KIND a type filter restricts to ('rule' / 'context'), or null for 'all'. A
// Rule-* filter shows ONLY rules; a Context-* filter shows ONLY context — the other
// kind is HIDDEN (not just faded). The specific tag / sub-type still FADES within.
function typeFilterKind() {
  const ft = curFilterType();
  return ft === 'all' ? null : ft.split(':')[0];
}
function visibleRules() {
  if (S.mode === 'relation') return S.rules;
  const k = typeFilterKind();
  return S.rules.filter(r => matchesSrc(r) && (!k || ruleKind(r) === k));
}

function setFilterSrc(v) {
  if (v === curFilterSrc()) return;
  S.filterSrc[S.mode] = v; localStorage.setItem('filterSrc.' + S.mode, v);   // per-mode
  afterFilterChange();
}
function setFilterType(v) {
  if (v === curFilterType()) return;
  S.filterType[S.mode] = v; localStorage.setItem('filterType.' + S.mode, v);   // per-mode
  afterFilterChange();
}
function afterFilterChange() {
  // the source filter HIDES — if the focused rule is now hidden, collapse it.
  if (S.focusedRuleId && !visibleRules().some(r => r.id === S.focusedRuleId)) S.focusedRuleId = null;
  renderViewer();        // source-hidden rules drop out of the document; type/search fade within
  renderRightPanel();    // re-render the rule list + filter selects
}
// Live keyword search: re-fade matching/non-matching cards IN PLACE (so the search box
// keeps focus while typing) and re-fade the document. Full re-renders honour S.search
// via matchesFilter too, so the two stay consistent.
let _searchScrollTimer = null;
function liveSearch(val) {
  S.search = val;
  const vis = visibleRules();         // the source-filtered set the list is showing
  const active = filterActive();
  let match = 0;
  for (const r of vis) {
    const ok = matchesFilter(r);
    if (ok) match++;
    document.getElementById('rl-' + r.id)?.classList.toggle('rl-faded', active && !ok);
  }
  const head = document.querySelector('.rl-head');
  if (head) head.textContent = `${vis.length} rule${vis.length === 1 ? '' : 's'}${active ? ` · ${match} match` : ''}`;
  renderViewer();        // fade the document highlights to match
  // jump the document to the FIRST matching rule (debounced so it settles after you pause
  // typing rather than lurching on every keystroke). rules are in document order.
  clearTimeout(_searchScrollTimer);
  _searchScrollTimer = setTimeout(() => {
    if (!(S.search || '').trim()) return;
    const first = visibleRules().find(r => matchesFilter(r) && getRuleSpans(r).length);
    if (first) scrollDocToNode(first.id);
  }, 180);
}

// blend rgba backgrounds — more overlap → more opaque
function parseRgba(s) {
  const m = s.match(/rgba?\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),?\s*(\d*\.?\d+)?\)/);
  return m ? [+m[1], +m[2], +m[3], m[4] !== undefined ? +m[4] : 1] : [200, 200, 220, 0.18];
}
function blendBgs(bgs) {
  if (bgs.length === 1) return bgs[0];
  const vals = bgs.map(parseRgba);
  const r = Math.round(vals.reduce((s, v) => s + v[0], 0) / vals.length);
  const g = Math.round(vals.reduce((s, v) => s + v[1], 0) / vals.length);
  const b = Math.round(vals.reduce((s, v) => s + v[2], 0) / vals.length);
  const baseA = vals.reduce((s, v) => s + v[3], 0) / vals.length;
  const a = Math.min(0.7, baseA * Math.sqrt(bgs.length));
  return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}
// Scale the alpha of an rgba() string (for the faded relation-mode highlights).
function fadeRgba(c, mult) {
  const v = parseRgba(c);
  return `rgba(${v[0]},${v[1]},${v[2]},${(v[3] * mult).toFixed(3)})`;
}

// ─── Resizable inspector panel ───────────────────────────────
function setInspWidth(px, save=true) {
  const layout = document.querySelector('.layout');
  const filePanel = document.querySelector('.file-panel');
  const maxW = layout && filePanel
    ? Math.max(INSP_MIN, Math.min(INSP_MAX, layout.clientWidth - filePanel.clientWidth - 340 - 8))
    : INSP_MAX;
  const width = clamp(Math.round(px), INSP_MIN, maxW);
  document.documentElement.style.setProperty('--insp-panel-width', `${width}px`);
  if (save) localStorage.setItem(INSP_WIDTH_KEY, String(width));
}
function initResizablePanel() {
  const saved = parseInt(localStorage.getItem(INSP_WIDTH_KEY) || '', 10);
  setInspWidth(Number.isNaN(saved) ? 420 : saved, false);
  const resizer = document.getElementById('panelResizer');
  const layout = document.querySelector('.layout');
  if (!resizer || !layout) return;
  function onMove(e) { setInspWidth(layout.getBoundingClientRect().right - e.clientX); }
  function stop() {
    document.body.classList.remove('resizing'); resizer.classList.remove('active');
    window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', stop);
  }
  resizer.addEventListener('mousedown', e => {
    e.preventDefault(); document.body.classList.add('resizing'); resizer.classList.add('active');
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', stop);
  });
}

// Load the per-mode judge config for the CURRENT annotator: reset to built-in
// defaults, then apply any saved config (judge panel + threshold + prompt). The
// config is stored as one JSON blob per mode under judge_cfg_<mode>. Re-run
// whenever the annotator changes (config is per-person).
async function loadJudgeSettings() {
  S.judgeCfg = { extract: defaultJudgeCfg('extract'), label: defaultJudgeCfg('label'), relation: defaultJudgeCfg('relation') };
  const aq = S.userName ? `?annotator=${encodeURIComponent(S.userName)}` : '';
  const settings = await api('/api/settings' + aq);
  if (settings && !settings.error) {
    for (const m of MODE_DEFS) {
      const raw = settings[m.key];
      if (!raw) continue;
      try {
        const cfg = JSON.parse(raw);
        const judges = Array.isArray(cfg.judges) && cfg.judges.length
          ? cfg.judges.map(j => ({ model: (typeof j === 'string' ? j : j.model) || 'sonar' }))
          : S.judgeCfg[m.id].judges;
        S.judgeCfg[m.id] = {
          judges,
          threshold: Math.max(1, Math.min(parseInt(cfg.threshold, 10) || Math.ceil(judges.length / 2), judges.length)),
          prompt: (typeof cfg.prompt === 'string' && cfg.prompt.trim()) ? cfg.prompt : m.deflt,
        };
      } catch (e) { /* leave default */ }
    }
  }
}

// Normalise /api/judge-data into S.judge, indexing votes by rule id and by edge
// so the vote rings + inspection panel can look them up cheaply.
function normalizeJudgeData(data) {
  S.judge = { extract: null, label: null, relation: null };
  if (!data || data.error) return;
  for (const mode of ['extract', 'label', 'relation']) {
    const info = data[mode];
    if (!info) continue;
    const byRule = {}, byEdge = {};
    for (const v of (info.votes || [])) {
      if (v.rule_id) (byRule[v.rule_id] = byRule[v.rule_id] || []).push(v);
      if (v.source_id && v.target_id) {
        const k = v.source_id + '::' + v.target_id;
        (byEdge[k] = byEdge[k] || []).push(v);
      }
    }
    info.byRule = byRule; info.byEdge = byEdge;
    S.judge[mode] = info;
  }
}

// ─── Boot ────────────────────────────────────────────────────
async function init() {
  initResizablePanel();
  await loadAnnotators();
  await loadJudgeSettings();
  const files = await api(filesUrl());
  if (files.error) { return; }
  S.allFiles = files;
  document.getElementById('fileCountLabel').textContent = files.length;
  renderFileList(files);
  renderModeToggle();   // reflect the persisted Rule/Relation mode
  // Click empty space (in the document margins or in the inspector) to de-select the
  // focused rule. Attached once — #viewerBody / #inspector persist across re-renders.
  document.getElementById('viewerBody').addEventListener('click', onViewerBodyClick);
  document.getElementById('inspector').addEventListener('click', onInspectorClick);
}

// De-select on a click in the viewer that isn't on a highlight and isn't a drag.
// Empty space INSIDE <pre> is handled by the <pre> click listener; this catches the
// gutter / area below the text. (Relation mode keeps its own click-to-pick UX.)
function onViewerBodyClick(e) {
  if (S.mode === 'relation' || S.editingRuleId) return;
  if (e.target.closest('.rule-hl')) return;       // highlight click focuses a rule
  if (e.target.closest('#contentPre')) return;    // empty space in <pre> handled there
  const sel = window.getSelection();
  if (sel && !sel.isCollapsed) return;            // a drag-select isn't a deselect click
  if (S.focusedRuleId) setFocusedRule(null);
}
// De-select on a click in the inspector that isn't on a rule card or a control.
function onInspectorClick(e) {
  if (S.mode === 'relation') return;
  if (e.target.closest('.rl-item, .rl-headbar, input, select, textarea, button')) return;
  if (S.focusedRuleId) setFocusedRule(null);
}

// ─── Annotators ──────────────────────────────────────────────
function filesUrl() {
  return S.userName ? `/api/files?annotator=${encodeURIComponent(S.userName)}` : '/api/files';
}

async function loadAnnotators() {
  const list = await api('/api/annotators');
  S.annotators = Array.isArray(list) ? list : [];
  // keep a sensible active annotator
  if (!S.userName && S.annotators.length) S.userName = S.annotators[0];
  renderAnnotatorSelect();
}

function renderAnnotatorSelect() {
  const sel = document.getElementById('annotatorSelect');
  if (!sel) return;
  const names = [...S.annotators];
  // include the current annotator even if they have no rows in this DB yet
  if (S.userName && !names.includes(S.userName)) names.unshift(S.userName);
  const opts = names.map(a =>
    `<option value="${esc(a)}"${a === S.userName ? ' selected' : ''}>${esc(a)}</option>`).join('');
  // A disabled placeholder when nobody is selected, so "＋ New annotator…" is ALWAYS a
  // distinct value — picking it always fires onchange (the old code pre-selected
  // __new__, so re-picking it did nothing and you couldn't add the first annotator).
  const placeholder = S.userName ? '' : `<option value="" disabled selected hidden>Select annotator…</option>`;
  sel.innerHTML = placeholder + opts + `<option value="__new__">＋ New annotator…</option>`;
}

async function onAnnotatorChange(v) {
  if (v === '__new__') { addAnnotatorPrompt(); return; }   // modal handles the rest
  S.userName = v;
  localStorage.setItem('annotatorName', S.userName);
  renderAnnotatorSelect();
  await refreshForAnnotator();
}

// Inline "new annotator" modal (replaces the native prompt(), which can be blocked /
// not fire). Creates the annotator, switches to it, and reloads their view.
function addAnnotatorPrompt() {
  document.getElementById('annotModal')?.remove();
  const modal = document.createElement('div');
  modal.id = 'annotModal'; modal.className = 'modal-backdrop';
  modal.innerHTML = `<div class="confirm-card">
    <div class="confirm-title">New annotator</div>
    <input id="annotNameInput" class="name-input" autocomplete="off" spellcheck="false"
      placeholder="Annotator name" style="width:100%;padding:8px 10px;margin:12px 0;font-size:14px;border:1px solid #c7c7e0;border-radius:8px">
    <div class="confirm-actions">
      <button class="confirm-cancel">Cancel</button>
      <button class="confirm-ok">Add</button>
    </div></div>`;
  document.body.appendChild(modal);
  const inp = modal.querySelector('#annotNameInput'); inp.focus();
  const close = (restore) => {
    modal.remove();
    document.removeEventListener('keydown', onKey, true);
    if (restore) renderAnnotatorSelect();   // put the <select> back to a valid value
  };
  const submit = async () => {
    const name = inp.value.trim();
    if (!name) { inp.focus(); return; }
    close(false);
    const res = await api('/api/annotators', 'POST', { name });
    if (res && res.error) { renderAnnotatorSelect(); return; }
    S.userName = name;
    if (!S.annotators.includes(name)) S.annotators.push(name);
    S.annotators.sort();
    localStorage.setItem('annotatorName', name);
    renderAnnotatorSelect();
    await refreshForAnnotator();
  };
  const onKey = e => {
    if (e.key === 'Escape') { e.stopPropagation(); close(true); }
    else if (e.key === 'Enter') { e.stopPropagation(); e.preventDefault(); submit(); }
  };
  document.addEventListener('keydown', onKey, true);
  modal.addEventListener('mousedown', e => { if (e.target === modal) close(true); });
  modal.querySelector('.confirm-cancel').onclick = () => close(true);
  modal.querySelector('.confirm-ok').onclick = submit;
}

// Reload the file list (counts) and the current file's rules for the active annotator.
async function refreshForAnnotator() {
  await loadJudgeSettings();   // judge prompts are per-annotator — load this person's
  const files = await api(filesUrl());
  if (Array.isArray(files)) { S.allFiles = files; renderFileList(files); }
  loadFileComment();   // the comment is per-annotator — refetch on switch
  if (S.currentFile) {
    const rules = await api(`/api/rules/${S.currentFile.id}?annotator=${encodeURIComponent(S.userName)}`);
    if (Array.isArray(rules)) { S.rules = rules; sortRules(); }
    S.focusedRuleId = null; S.selection = null; S.toolMode = false;
    document.getElementById('addBtn').disabled = true;
    document.getElementById('judgeBtn').classList.remove('active');
    renderJudgeModal();
    renderViewer();
    renderRightPanel();
  }
}

// ─── File list ───────────────────────────────────────────────
function renderFileList(files) {
  const el = document.getElementById('fileList');
  if (!files.length) { el.innerHTML = '<div class="insp-empty">No files</div>'; return; }
  const idxMap = {};
  S.allFiles.forEach((f, i) => idxMap[f.id] = i + 1);
  el.innerHTML = files.map(f => {
    const name = f.repo_name ? f.repo_name.split('/').pop()
      : (f.source_url || f.id).split('/').pop() || f.id;
    // Count badges: red ✎ = human labels; machine ⚖ = blue for LLM extraction, GREEN
    // once a "revise" pass has been run on the file.
    const badge =
      ((f.hand_count || 0) ? `<span class="badge hand" title="${f.hand_count} human label${f.hand_count === 1 ? '' : 's'}">✎ ${f.hand_count}</span>` : '') +
      ((f.llm_count  || 0) ? `<span class="badge ${f.revised ? 'revised' : 'llm'}" title="${f.llm_count} ${f.revised ? 'revised' : 'LLM'} label${f.llm_count === 1 ? '' : 's'}">⚖ ${f.llm_count}</span>` : '');
    const active = S.currentFile?.id === f.id ? ' active' : '';
    return `<div class="file-item${active}" id="fi-${f.id}" onclick="selectFile('${f.id}')">
      <div class="file-idx">${idxMap[f.id] || '?'}</div>
      <div class="file-info">
        <div class="file-name" title="${esc(f.source_url || f.id)}">${esc(name)}</div>
        <div class="file-meta">
          <span>${esc(f.file_type || '—')}</span>
          <span>${fmtSize(f.content_len)}</span>
          ${badge}
        </div>
      </div>
    </div>`;
  }).join('');
}

function filterFiles(q) {
  q = (q || '').toLowerCase();
  renderFileList(q
    ? S.allFiles.filter(f =>
        (f.repo_name || '').toLowerCase().includes(q) ||
        (f.source_url || '').toLowerCase().includes(q) ||
        (f.file_type || '').toLowerCase().includes(q))
    : S.allFiles);
}

// ─── Select file ─────────────────────────────────────────────
async function selectFile(id) {
  if (S.judgeRunning) return;   // locked while a judge run is in flight
  const aq = S.userName ? `?annotator=${encodeURIComponent(S.userName)}` : '';
  const [file, rules, relations, jdata] = await Promise.all([
    api(`/api/file/${id}`), api(`/api/rules/${id}${aq}`), api(`/api/relations/${id}${aq}`),
    api(`/api/judge-data/${id}${aq}`),
  ]);
  if (file.error) return;
  S.currentFile = file; S.rules = rules; sortRules();
  S.relations = Array.isArray(relations) ? relations : [];
  normalizeJudgeData(jdata);
  S.relSource = S.relTarget = S.focusedRelId = null;
  S.selection = null; S.focusedRuleId = null; S.toolMode = false; S.inspectorOpen = false; S.search = '';
  S.editingRuleId = null; S.editSelection = null;
  S.commentMode = false; S.fileComment = '';
  document.getElementById('addBtn').disabled = true;
  document.getElementById('judgeBtn').classList.remove('active');
  document.getElementById('commentBtn').classList.remove('active');
  setSelInfo(null);
  renderFileList(S.allFiles);
  renderViewerHead(file);
  renderViewer();
  document.getElementById('viewerBody').scrollTop = 0;   // new file → start at the top
  renderRightPanel();
  renderJudgeModal();     // close the judge modal if it was open
  renderCommentModal();   // close the comment panel if it was open
  renderModeToggle();     // apply +Add visibility for the current mode
  renderRelBuild();       // hide the relation-build bar for the new file
  loadFileComment();      // async; fills the green has-comment indicator
}

function renderViewerHead(file) {
  const title = file.repo_name
    ? `${file.repo_name}  /  ${(file.source_url || '').split('/').pop()}`
    : (file.source_url || file.id).split('/').pop();
  document.getElementById('viewerHead').innerHTML = `
    <span class="viewer-title" title="${esc(file.source_url || '')}">${esc(title)}</span>
    ${file.source_url ? `<a href="${esc(file.source_url)}" target="_blank">↗ source</a>` : ''}
    <span style="font-size:11px;color:#aaa;flex-shrink:0">${fmtSize(file.content_len)}</span>
  `;
}

// ─── Char ranges ─────────────────────────────────────────────
function getRuleCharRange(rule) {
  const content = S.currentFile?.content;
  if (!content) return null;
  const cs = rule.char_start, ce = rule.char_end;
  if (cs != null && cs !== '' && ce != null && ce !== '') {
    const a = Number(cs), b = Number(ce);
    if (Number.isFinite(a) && Number.isFinite(b) && b > a)
      return [clamp(Math.min(a, b), 0, content.length), clamp(Math.max(a, b), 0, content.length)];
  }
  // fallback: derive from line range
  const lsRaw = rule.line_start;
  if (lsRaw == null || lsRaw === '') return null;
  const leRaw = rule.line_end != null && rule.line_end !== '' ? rule.line_end : lsRaw;
  const ls = Math.max(1, Number(lsRaw)), le = Math.max(ls, Number(leRaw));
  if (!Number.isFinite(ls) || !Number.isFinite(le)) return null;
  const lines = content.split('\n');
  let off = 0;
  for (let i = 0; i < ls - 1 && i < lines.length; i++) off += lines[i].length + 1;
  const start = off;
  let end = off;
  for (let i = ls - 1; i < le && i < lines.length; i++) end += lines[i].length + 1;
  end = Math.min(end, content.length);
  return [start, Math.max(start, end)];
}

// ─── Viewer (document or tool) ───────────────────────────────
function renderViewer() {
  const body = document.getElementById('viewerBody');
  if (!S.currentFile) { body.innerHTML = '<div class="viewer-empty">← pick a file</div>'; return; }

  // The document text is identical across re-renders (only highlight colours change),
  // so preserve the scroll position — switching Rule/Relation mode, picking nodes,
  // adding rules etc. should NOT jump the body back to the top.
  const prevScroll = body.scrollTop;
  const content = S.currentFile.content;
  const lineCount = (content.match(/\n/g) || []).length + 1;
  const lineNums = Array.from({ length: lineCount }, (_, i) =>
    `<span class="line-num">${i + 1}</span>`).join('');
  body.innerHTML = `
    <div class="line-nums">${lineNums}</div>
    <pre class="content-pre" id="contentPre"></pre>`;
  document.getElementById('contentPre').innerHTML = renderContent(content, visibleRules());
  body.scrollTop = prevScroll;   // restore after the rebuild reset it to 0
  const pre = document.getElementById('contentPre');
  // NOTE: drag-select → Add is captured by a document-level 'mouseup' (added once
  // below), so it still works when the drag is released outside the <pre>.
  pre.addEventListener('click', e => {
    const span = e.target.closest?.('.rule-hl');
    if (S.mode === 'relation') {
      if (!span) return;
      // plain click = source, ⌘-click (Ctrl on Win) a second rule = target.
      const isTarget = e.metaKey || e.ctrlKey;
      const sel = window.getSelection();
      if (!isTarget && sel && !sel.isCollapsed) return;   // a real drag-select isn't a source pick
      // clear any stray native selection so the pick lands
      if (isTarget) window.getSelection()?.removeAllRanges();
      pickRelNode(span, isTarget);
      return;
    }
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) return;   // rule mode: text selection handled on mouseup
    // While editing a rule's text, a plain click (no drag = "unselect") aborts the edit.
    if (S.editingRuleId) { cancelRuleEdit(); return; }
    if (span) focusSpan(span);
    else {                                 // clicking empty space clears a pending
      clearSelection();                    // selection AND de-selects the focused rule
      if (S.focusedRuleId) setFocusedRule(null);
    }
  });
  pre.addEventListener('mouseover', e => {
    const el = e.target.closest?.('.rule-hl');
    if (el && !el.dataset.rids.split(',').includes(S.focusedRuleId)) el.style.background = el.dataset.hov;
  });
  pre.addEventListener('mouseout', e => {
    const el = e.target.closest?.('.rule-hl');
    if (el && !el.dataset.rids.split(',').includes(S.focusedRuleId)) el.style.background = el.dataset.bg;
  });
}

// One rule can map to several highlight segments — e.g. a quote that spans a
// paragraph gap ("…respond like this: \"Heads up…") matches in two pieces, and
// both get highlighted together under the same rule.
function getRuleSpans(rule) {
  const content = S.currentFile?.content;
  if (!content) return [];
  if (rule._spans && rule._spansFor === content) return rule._spans;
  const rt = (rule.rule_text || '').trim();
  let spans = null;
  // If the stored char range EXACTLY contains this rule's text, use it — this pins the
  // rule to ITS OWN occurrence, so two rules with identical text highlight different
  // spots (instead of both snapping to the first text match).
  const cr = getRuleCharRange(rule);
  if (cr && content.slice(cr[0], cr[1]).trim() === rt && rt) spans = [cr];
  // Otherwise locate by text (handles multi-segment quotes / drifted offsets), then
  // fall back to the stored range as a last resort.
  if (!spans) {
    spans = locateSpans(content, rt);
    if (!spans.length && cr) spans = [cr];
  }
  rule._spans = spans; rule._spansFor = content;
  return spans;
}

// Greedy multi-segment matcher: anchor the longest matching prefix, then keep
// matching the remainder (skipping separator junk) to recover non-contiguous
// segments of the same quote.
function locateSpans(content, rt) {
  if (!rt) return [];
  const cl = content.toLowerCase();
  let idx = content.indexOf(rt);                 // fast path: contiguous match
  if (idx === -1) idx = cl.indexOf(rt.toLowerCase());
  if (idx >= 0) return [[idx, idx + rt.length]];
  let rem = rt.toLowerCase(), from = 0, guard = 0;
  const spans = [];
  while (rem.length >= 6 && guard++ < 60) {
    let lo = 6, hi = rem.length, bestLen = 0, bestIdx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const j = cl.indexOf(rem.slice(0, mid), from);
      if (j !== -1) { bestLen = mid; bestIdx = j; lo = mid + 1; }
      else hi = mid - 1;
    }
    if (bestIdx < 0 || bestLen < 6) { rem = rem.slice(1); continue; }
    spans.push([bestIdx, bestIdx + bestLen]);
    from = bestIdx + bestLen;
    rem = rem.slice(bestLen).replace(/^[\s"'>\-—:.,()]+/, '');  // drop separator junk
  }
  spans.sort((a, b) => a[0] - b[0]);             // merge touching/overlapping
  const merged = [];
  for (const s of spans) {
    const last = merged[merged.length - 1];
    if (last && s[0] <= last[1] + 2) last[1] = Math.max(last[1], s[1]);
    else merged.push([s[0], s[1]]);
  }
  return merged;
}

// Extract-mode palette: by kind (rule green / context yellow), plus red for human
// spans. solid = accepted (≥ threshold); light = candidate; act = focused; hov = hover.
const EXTRACT_COL = {
  // `light` = candidate (sub-threshold) spans — kept faint (~2/3 of the prior alpha)
  // so accepted (solid) spans clearly stand out.
  rule:    { solid: 'rgba(34,197,94,.30)',  light: 'rgba(34,197,94,.087)', act: 'rgba(34,197,94,.52)',  hov: 'rgba(34,197,94,.42)'  },
  context: { solid: 'rgba(234,179,8,.34)',  light: 'rgba(234,179,8,.10)',  act: 'rgba(234,179,8,.56)',  hov: 'rgba(234,179,8,.46)'  },
  hand:    { solid: 'rgba(239,68,68,.30)',  light: 'rgba(239,68,68,.10)',  act: 'rgba(239,68,68,.54)',  hov: 'rgba(239,68,68,.44)'  },
};
function extractColForRule(r) {
  if (r.source === 'hand') return EXTRACT_COL.hand;
  return EXTRACT_COL[ruleKind(r)] || EXTRACT_COL.rule;
}
// Left-edge bar colour for a rule-list card: Extract → kind (green/yellow/red),
// Label → deontic tag / context type.
function listBarColor(r) {
  if (S.mode === 'extract') { const c = extractColForRule(r); return { bdr: c.act, bg: c.light }; }
  const c = typeColorForRule(r);
  return { bdr: c.act, bg: c.bg };
}

// letter-based highlighting: segment text on every rule boundary, styled per mode.
function renderContent(content, rules) {
  const ivs = [];
  for (const r of rules) {
    for (const [a, b] of getRuleSpans(r)) if (b > a) ivs.push({ a, b, r });
  }
  // Extract: sub-threshold judge clusters render as light, non-clickable candidates.
  const cands = (S.mode === 'extract' && S.judge.extract) ? (S.judge.extract.candidates || []) : [];
  for (const c of cands) {
    if (c.char_start != null && c.char_end != null && c.char_end > c.char_start)
      ivs.push({ a: c.char_start, b: c.char_end, cand: c });
  }
  if (!ivs.length) return escHtml(content);
  const pts = new Set([0, content.length]);
  for (const iv of ivs) { pts.add(iv.a); pts.add(iv.b); }
  const sorted = [...pts].filter(p => p >= 0 && p <= content.length).sort((x, y) => x - y);
  let html = '';
  for (let i = 0; i < sorted.length - 1; i++) {
    const p = sorted[i], q = sorted[i + 1];
    if (q <= p) continue;
    const seg = content.slice(p, q);
    const cover = ivs.filter(iv => iv.a <= p && iv.b >= q);
    if (!cover.length) { html += escHtml(seg); continue; }
    const ruleCover = cover.filter(c => c.r);
    const candCover = cover.filter(c => c.cand);

    // ── A real rule covers this segment: solid + clickable ──
    if (ruleCover.length) {
      const rids = ruleCover.map(c => c.r.id);
      const title = escAttr(ruleCover.map(c => extractorOf(c.r) + ': ' + c.r.rule_text.slice(0, 60)).join(' | '));
      if (S.mode === 'relation') {
        const [hs, ht] = relHlPair();
        const cs = hs && rids.includes(hs), ct = ht && rids.includes(ht);
        const full = blendBgs(ruleCover.map(c => typeColorForRule(c.r).act));
        const bg = (cs || ct) ? full : fadeRgba(blendBgs(ruleCover.map(c => typeColorForRule(c.r).bg)), 0.6);
        const uline = (cs && ct) ? ' hl-both' : cs ? ' hl-src' : ct ? ' hl-tgt' : '';
        html += `<span class="rule-hl rel-hl${uline}" data-rids="${rids.join(',')}"`
          + ` data-bg="${bg}" data-hov="${bg}" data-act="${bg}" style="background:${bg}" title="${title}">${escHtml(seg)}</span>`;
        continue;
      }
      let bg, hov, act;
      if (S.mode === 'extract') {         // colour by kind (green/yellow) or red for human
        bg  = blendBgs(ruleCover.map(c => extractColForRule(c.r).solid));
        hov = blendBgs(ruleCover.map(c => extractColForRule(c.r).hov));
        act = blendBgs(ruleCover.map(c => extractColForRule(c.r).act));
      } else {                            // label: colour by tag/context — LIGHT until selected
        bg  = blendBgs(ruleCover.map(c => fadeRgba(typeColorForRule(c.r).bg, 0.5)));  // not selected → light
        hov = blendBgs(ruleCover.map(c => typeColorForRule(c.r).bg));                 // medium on hover
        act = blendBgs(ruleCover.map(c => typeColorForRule(c.r).act));               // selected → full
      }
      const focused = rids.includes(S.focusedRuleId);
      const editing = S.editingRuleId && rids.includes(S.editingRuleId);
      const dim = filterActive() && !ruleCover.some(c => matchesFilter(c.r));
      const rest = editing ? 'rgba(150,150,170,0.28)' : (dim ? fadeRgba(bg, 0.3) : bg);
      html += `<span class="rule-hl${focused && !editing ? ' focused' : ''}" data-rids="${rids.join(',')}"`
        + ` data-bg="${rest}" data-hov="${editing || dim ? rest : hov}" data-act="${editing ? rest : act}"`
        + ` style="background:${focused && !editing ? act : rest}" title="${title}">${escHtml(seg)}</span>`;
      continue;
    }
    // ── Only candidate spans here (Extract): light, not clickable ──
    if (candCover.length) {
      const kind = candCover[0].cand.kind === 'context' ? 'context' : 'rule';
      const nj = Math.max(...candCover.map(c => c.cand.vote_count || 1));
      const title = escAttr(`candidate ${kind} · ${nj} judge${nj === 1 ? '' : 's'} (below threshold)`);
      html += `<span class="cand-hl" style="background:${EXTRACT_COL[kind].light}" title="${title}">${escHtml(seg)}</span>`;
      continue;
    }
    html += escHtml(seg);
  }
  return html;
}

// ─── Text selection / click-to-focus ─────────────────────────
// Document-level so a drag released anywhere (incl. outside the <pre>, past the end
// of a line) still registers. Relation mode never adds rules; collapsed clicks are
// handled by the <pre> click listener.
function onViewerMouseUp() {
  if (S.mode !== 'extract' || !S.currentFile) return;   // spans are created in Extract only
  const pre = document.getElementById('contentPre');
  if (!pre) return;
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;          // not a drag-selection
  const offsets = getSelectionOffsets(pre);
  if (!offsets) return;                          // selection isn't in the document
  const preview = offsets.text.slice(0, 55).replace(/\s+/g, ' ');
  // Editing a rule's text: the drag re-defines THIS rule, not a new one.
  if (S.editingRuleId) {
    S.editSelection = offsets;
    setSelInfo(`new text: "${preview}${offsets.text.length > 55 ? '…' : ''}" — <kbd>↵</kbd> save · <kbd>Esc</kbd> cancel`);
    return;
  }
  S.selection = offsets;
  setSelInfo(`"${preview}${offsets.text.length > 55 ? '…' : ''}" (${offsets.end - offsets.start} chars) — <kbd>⌘↵</kbd>`);
  document.getElementById('addBtn').disabled = false;
}
document.addEventListener('mouseup', onViewerMouseUp);

// ── Edit a rule's TEXT by re-selecting (press E on a focused rule) ──
function startRuleEdit() {
  if (!S.focusedRuleId) return;
  S.editingRuleId = S.focusedRuleId;
  S.editSelection = null;
  window.getSelection()?.removeAllRanges();
  document.getElementById('addBtn').disabled = true;
  setSelInfo('editing rule text — drag-select the new text · <kbd>↵</kbd> save · <kbd>Esc</kbd> cancel');
  renderViewer();          // grey out the rule's current highlight
  renderKbBar();
}
function commitRuleEdit() {
  const r = ruleById(S.editingRuleId);
  const sel = S.editSelection;
  if (r && sel && sel.text.trim()) {            // new text picked → save it to the rule
    const content = S.currentFile.content;
    r.rule_text  = sel.text.trim();
    r.char_start = sel.start; r.char_end = sel.end;
    r.line_start = content.slice(0, sel.start).split('\n').length;
    r.line_end   = content.slice(0, sel.end).split('\n').length;
    r._spans = null;                            // force the highlight to re-derive
    const patch = { rule_text: r.rule_text, char_start: r.char_start, char_end: r.char_end,
                    line_start: r.line_start, line_end: r.line_end };
    if (markReviewed(r)) patch.reviewed = true; // editing an LLM rule = human review
    api(`/api/rules/${r.id}`, 'PATCH', patch);
  }
  endRuleEdit();                                 // no selection → abort (just exits)
}
function cancelRuleEdit() { endRuleEdit(); }
function endRuleEdit() {
  S.editingRuleId = null; S.editSelection = null;
  window.getSelection()?.removeAllRanges();
  setSelInfo(null);
  sortRules();             // position may have changed
  renderViewer(); renderRuleList(); renderKbBar();
}

function focusSpan(span) {
  const rids = span.dataset.rids.split(',');
  let id = rids[0];
  const cur = rids.indexOf(S.focusedRuleId);
  if (cur >= 0 && rids.length > 1) id = rids[(cur + 1) % rids.length];
  clearSelection();
  setFocusedRule(id);
}

function clearSelection() {
  S.selection = null;
  document.getElementById('addBtn').disabled = true;
  setSelInfo(null);
}

function setSelInfo(html) {
  const el = document.getElementById('selInfo');
  if (el) el.innerHTML = html || '';
}

function boundaryOffset(container, node, offset) {
  const r = document.createRange();
  r.selectNodeContents(container);
  r.setEnd(node, offset);
  return r.toString().length;
}
function getSelectionOffsets(container) {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
  let range = sel.getRangeAt(0);
  const full = document.createRange();
  full.selectNodeContents(container);
  // The selection must overlap the document container at all…
  if (range.compareBoundaryPoints(Range.START_TO_END, full) < 0 ||   // ends before container starts
      range.compareBoundaryPoints(Range.END_TO_START, full) > 0) {    // starts after container ends
    return null;
  }
  // …then clamp it to the container so a drag that overshoots the <pre> (past the
  // end of a line, into the gutter, etc.) still captures the in-document portion.
  range = range.cloneRange();
  if (range.compareBoundaryPoints(Range.START_TO_START, full) < 0) range.setStart(full.startContainer, full.startOffset);
  if (range.compareBoundaryPoints(Range.END_TO_END, full) > 0) range.setEnd(full.endContainer, full.endOffset);
  const text = range.toString();
  if (!text.trim()) return null;
  const start = boundaryOffset(container, range.startContainer, range.startOffset);
  const end = boundaryOffset(container, range.endContainer, range.endOffset);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return { start, end: Math.max(start, end), text };
}

async function addHandRule() {
  if (!S.selection || !S.currentFile) return;
  const { start, end, text } = S.selection;
  const content = S.currentFile.content;
  const line_start = content.slice(0, start).split('\n').length;
  const line_end   = content.slice(0, end).split('\n').length;
  const saved = await api('/api/rules', 'POST', {
    file_id: S.currentFile.id, rule_text: text.trim(),
    char_start: start, char_end: end, line_start, line_end,
    extracted_by: S.userName || 'unknown',
    annotator: S.userName || 'unknown',
  });
  if (saved.error) return;
  S.rules.push(saved); sortRules();
  clearSelection();
  window.getSelection()?.removeAllRanges();
  renderViewer();
  refreshFileBadge(S.currentFile.id);
  setFocusedRule(saved.id);
}

// ─── Focus / navigation ──────────────────────────────────────
// Order rules by where they appear in the document (line, then char).
function sortRules() {
  const k = v => (v == null || v === '' ? Infinity : Number(v));
  S.rules.sort((a, b) =>
    (k(a.line_start) - k(b.line_start)) ||
    (k(a.char_start) - k(b.char_start)));
}

function paintFocus() {
  document.querySelectorAll('.rule-hl').forEach(el => {
    const ids = el.dataset.rids.split(',');
    const on = ids.includes(S.focusedRuleId);
    el.style.background = on ? el.dataset.act : el.dataset.bg;
    el.classList.toggle('focused', on);
  });
}

function setFocusedRule(id) {
  S.focusedRuleId = id;
  paintFocus();
  renderRuleList();
  if (id) {
    document.getElementById('rl-' + id)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    // center the rule's highlighted text in the document viewer
    const span = [...document.querySelectorAll('.rule-hl')]
      .find(el => el.dataset.rids.split(',').includes(id));
    span?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function renderRightPanel() {
  if (S.mode === 'relation') renderRelationList();
  else renderRuleList();
}

// ─── Extract / Label / Relation mode ─────────────────────────
// Per-mode document highlight legend.
const MODE_LEGEND = {
  extract: `<span class="tl-label">Extract:</span>
    <span class="tl-item"><i style="background:#22c55e"></i>Rule</span>
    <span class="tl-item"><i style="background:#eab308"></i>Context</span>
    <span class="tl-item"><i style="background:#ef4444"></i>You</span>
    <span class="tl-item" style="opacity:.6"><i style="background:#9aa6b2"></i>candidate (light)</span>`,
  label: `<span class="tl-label">Colours:</span>
    <span class="tl-item"><i style="background:#cf4436"></i>Prohibition</span>
    <span class="tl-item"><i style="background:#2f6cdf"></i>Prescription</span>
    <span class="tl-item"><i style="background:#1c9b54"></i>Permission</span>
    <span class="tl-item"><i style="background:#8a4fd0"></i>Preference</span>
    <span class="tl-item"><i style="background:#6b7196"></i>Context</span>`,
  relation: `<span class="tl-label">Colours:</span>
    <span class="tl-item"><i style="background:#cf4436"></i>Prohibition</span>
    <span class="tl-item"><i style="background:#2f6cdf"></i>Prescription</span>
    <span class="tl-item"><i style="background:#1c9b54"></i>Permission</span>
    <span class="tl-item"><i style="background:#8a4fd0"></i>Preference</span>
    <span class="tl-item"><i style="background:#6b7196"></i>Context</span>`,
};
function renderModeToggle() {
  const t = document.getElementById('modeToggle');
  if (t) t.className = 'mode-toggle m-' + S.mode;
  // + Add (select text → new span) only makes sense in Extract — that's where
  // rules are created. Label tags them; Relation links them.
  const add = document.getElementById('addBtn');
  if (add) add.style.display = S.mode === 'extract' ? '' : 'none';
  // Per-mode colour legend strip.
  const leg = document.getElementById('typeLegend');
  if (leg) { leg.innerHTML = MODE_LEGEND[S.mode] || ''; leg.style.display = 'flex'; }
  renderKbBar();
}

// The shortcut hint bar — reflects the active mode + focus state.
function renderKbBar() {
  const kb = document.getElementById('kbBar');
  if (!kb) return;
  if (S.mode === 'relation') {
    kb.innerHTML = `<span><kbd>click</kbd> source</span><span><kbd>⌘+click</kbd> target</span><span><kbd>⌘↵</kbd> add</span><span><kbd>Del</kbd> cancel</span>`;
    return;
  }
  if (S.editingRuleId) {   // re-selecting a rule's text
    kb.innerHTML = `<span>drag to re-select text</span><span><kbd>↵</kbd> save</span><span><kbd>Esc</kbd> cancel</span>`;
    return;
  }
  const r = S.focusedRuleId && S.rules.find(x => x.id === S.focusedRuleId);
  const segs = [];
  if (S.mode === 'label') {                    // Label: a/s/d/f set the deontic tag
    segs.push(r && ruleKind(r) === 'rule'
      ? `<span><kbd>a</kbd><kbd>s</kbd><kbd>d</kbd><kbd>f</kbd> tag</span>`
      : `<span>pick a rule to tag</span>`);
  } else {                                     // Extract: select text → add a span
    segs.push(`<span><kbd>⌘↵</kbd> add span</span>`);
    if (r) segs.push(`<span><kbd>e</kbd> edit/extend</span>`);
  }
  segs.push(`<span><kbd>j</kbd><kbd>k</kbd> nav</span>`);
  if (r) segs.push(`<span><kbd>⌫</kbd> del</span>`);
  segs.push(`<span><kbd>Esc</kbd> clear</span>`);
  kb.innerHTML = segs.join('');
}
function setMode(m) {
  if (m === S.mode || !['extract', 'label', 'relation'].includes(m)) return;
  S.mode = m;
  localStorage.setItem('annotatorMode', m);
  S.relSource = S.relTarget = S.focusedRelId = null;   // drop any in-progress/focused edge
  S.focusedRuleId = null;                  // collapse rule inspector when leaving
  S.editingRuleId = null; S.editSelection = null;      // abort any in-progress text edit
  renderModeToggle();
  setSelInfo(null);
  clearSelection();
  renderViewer();
  renderRightPanel();
  renderRelBuild();
}

// The (source,target) pair currently driving the document highlight: the edge
// being built if any, else the relation focused in the list.
function relHlPair() {
  if (S.relSource || S.relTarget) return [S.relSource, S.relTarget];
  const rel = S.relations.find(r => r.id === S.focusedRelId);
  return rel ? [rel.source_id, rel.target_id] : [null, null];
}
function ruleById(id) { return S.rules.find(x => x.id === id); }
// The label to show for an edge: the user's label takes priority; fall back to
// the LLM's suggested label. (Both are stored so they can be contrasted.)
function effectiveType(rel) { return rel.relation_type || rel.llm_relation_type || null; }
function ruleShort(id, n = 46) {
  const r = ruleById(id);
  if (!r) return '(missing rule)';
  const t = (r.rule_text || '').replace(/\s+/g, ' ').trim();
  return t.length > n ? t.slice(0, n) + '…' : t;
}
// A clicked highlight may cover several rules — pick the most specific (shortest).
function pickRuleFromSpan(span) {
  const rids = (span.dataset.rids || '').split(',').filter(Boolean);
  if (rids.length <= 1) return rids[0] || null;
  let best = rids[0], bestLen = Infinity;
  for (const id of rids) {
    const r = ruleById(id);
    const len = r ? (r.rule_text || '').length : Infinity;
    if (len < bestLen) { bestLen = len; best = id; }
  }
  return best;
}
function pickRelNode(span, isTarget) {
  const id = pickRuleFromSpan(span);
  if (!id) return;
  // ⌘-click only means "target" once a source exists; otherwise it picks the source.
  if (isTarget && S.relSource) {
    if (id === S.relSource) return;        // target can't equal source
    S.relTarget = id;
  } else {
    S.relSource = id;
    if (S.relTarget === id) S.relTarget = null;
  }
  S.focusedRelId = null;                    // building a new edge, not viewing one
  renderViewer();
  renderRelBuild();
}
function swapRel() {
  if (!S.relTarget) return;
  const s = S.relSource; S.relSource = S.relTarget; S.relTarget = s;
  renderViewer(); renderRelBuild();
}
function clearRelBuild() {
  S.relSource = S.relTarget = null;
  renderViewer(); renderRelBuild();
}
async function addRelation() {
  if (!S.relSource || !S.relTarget) return;
  const res = await api('/api/relations', 'POST', {
    file_id: S.currentFile.id, source_id: S.relSource, target_id: S.relTarget,
    relation_type: S.relType, annotator: S.userName || 'unknown',
  });
  if (res.error) { setSelInfo(`<span class="sel-err">${esc(res.error)}</span>`); return; }
  if (!S.relations.some(r => r.id === res.id)) S.relations.push(res);
  S.relSource = S.relTarget = null;
  S.focusedRelId = res.id;
  renderViewer(); renderRelBuild(); renderRightPanel();
}
// Floating bar shown while picking the two endpoints of a new relation.
function renderRelBuild() {
  const bar = document.getElementById('relBuild');
  if (!bar) return;
  if (S.mode !== 'relation' || !S.relSource) { bar.style.display = 'none'; bar.innerHTML = ''; return; }
  const tgt = S.relTarget;
  const opts = REL_TYPES.map(t => `<option value="${esc(t)}"${t === S.relType ? ' selected' : ''}>${esc(t)}</option>`).join('');
  bar.style.display = 'flex';
  bar.innerHTML = `
    <span class="rb-pill s" title="source" style="background:${typeBadgeColor(ruleById(S.relSource))}">${esc(ruleShort(S.relSource, 30))}</span>
    <button class="rb-ico" title="swap source/target" onclick="swapRel()">⇄</button>
    ${tgt ? `<span class="rb-pill t" title="target" style="background:${typeBadgeColor(ruleById(tgt))}">${esc(ruleShort(tgt, 30))}</span>`
          : `<span class="rb-hint">⌘-click a target rule…</span>`}
    <select class="rb-type" onchange="S.relType=this.value"${tgt ? '' : ' disabled'}>${opts}</select>
    <button class="rb-add" title="Add relation (⌘↵)" onclick="addRelation()"${tgt ? '' : ' disabled'}>＋ Add relation <span class="rb-kbd">⌘↵</span></button>
    <button class="rb-ico" title="cancel (Delete)" onclick="clearRelBuild()">✕</button>`;
}

// ─── Relation list (right panel in relation mode) ────────────
function focusRelation(id) {
  S.focusedRelId = (S.focusedRelId === id) ? null : id;
  S.relSource = S.relTarget = null;
  renderViewer(); renderRightPanel();
  if (S.focusedRelId) {
    // the pair is now highlighted (relHlPair) — scroll the document so the source
    // (and ideally the target) comes into view. scrollIntoView is a no-op on the
    // flex .viewer-body, so use the direct-scrollTop helper.
    const rel = S.relations.find(r => r.id === id);
    if (rel) scrollDocToRelation(rel);
  }
}
// Scroll the document viewer to a node's highlight (used from the relation detail).
// scrollIntoView doesn't move the flex .viewer-body reliably, so set scrollTop directly.
function scrollDocToNode(id) {
  const span = [...document.querySelectorAll('.rule-hl')].find(el => el.dataset.rids.split(',').includes(id));
  if (!span) return;
  const vb = document.querySelector('.viewer-body');
  if (!vb) { span.scrollIntoView({ behavior: 'smooth', block: 'center' }); return; }
  const sr = span.getBoundingClientRect(), br = vb.getBoundingClientRect();
  const h = vb.clientHeight || br.height || 600;
  vb.scrollTop = Math.max(0, vb.scrollTop + (sr.top - br.top) - h / 2 + sr.height / 2);
}
// Bring a whole relation into view: if both endpoints fit on screen, centre their
// span together; otherwise fall back to centring the source.
function scrollDocToRelation(rel) {
  const findSpan = id => [...document.querySelectorAll('.rule-hl')].find(el => el.dataset.rids.split(',').includes(id));
  const ss = findSpan(rel.source_id), ts = findSpan(rel.target_id);
  const vb = document.querySelector('.viewer-body');
  if (!vb || !ss) { scrollDocToNode(rel.source_id); return; }
  const br = vb.getBoundingClientRect();
  const h = vb.clientHeight || br.height || 600;
  const sr = ss.getBoundingClientRect();
  if (ts) {
    const tr = ts.getBoundingClientRect();
    const top = Math.min(sr.top, tr.top), bot = Math.max(sr.bottom, tr.bottom);
    if (bot - top <= h - 24) {                 // both fit — centre the pair
      vb.scrollTop = Math.max(0, vb.scrollTop + ((top + bot) / 2 - br.top) - h / 2);
      return;
    }
  }
  vb.scrollTop = Math.max(0, vb.scrollTop + (sr.top - br.top) - h / 2 + sr.height / 2);
}
async function deleteRelation(evt, id) {
  evt?.stopPropagation();
  await api(`/api/relations/${id}`, 'DELETE');
  S.relations = S.relations.filter(r => r.id !== id);
  if (S.focusedRelId === id) S.focusedRelId = null;
  renderViewer(); renderRightPanel();
}
function deleteAllRelations() {
  const n = S.relations.length;
  if (!n || !S.currentFile) return;
  confirmDanger(
    'Delete all relations?',
    `This permanently removes all ${n} relation${n === 1 ? '' : 's'} in this file. This can't be undone.`,
    'Delete all',
    async () => {
      const q = `?file_id=${encodeURIComponent(S.currentFile.id)}` +
                (S.userName ? `&annotator=${encodeURIComponent(S.userName)}` : '');
      await api('/api/relations' + q, 'DELETE');
      S.relations = []; S.focusedRelId = S.relSource = S.relTarget = null;
      renderViewer(); renderRightPanel(); renderRelBuild();
    });
}
// Swap an edge's source ↔ target (click the arrow in the relation row).
async function swapRelation(evt, id) {
  evt?.stopPropagation();
  const rel = S.relations.find(r => r.id === id);
  if (!rel) return;
  const ns = rel.target_id, nt = rel.source_id;
  rel.source_id = ns; rel.target_id = nt;          // optimistic
  renderViewer(); renderRightPanel();
  await api(`/api/relations/${id}`, 'PATCH', { source_id: ns, target_id: nt });
}
// A styled danger confirmation (replaces the native confirm()).
function confirmDanger(title, message, confirmLabel, onConfirm) {
  document.getElementById('confirmModal')?.remove();
  const modal = document.createElement('div');
  modal.id = 'confirmModal'; modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="confirm-card">
      <div class="confirm-title">${esc(title)}</div>
      <div class="confirm-msg">${esc(message)}</div>
      <div class="confirm-actions">
        <button class="confirm-cancel">Cancel</button>
        <button class="confirm-ok">${esc(confirmLabel)}</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
  const close = () => { modal.remove(); document.removeEventListener('keydown', onKey, true); };
  const onKey = e => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
  document.addEventListener('keydown', onKey, true);
  modal.addEventListener('mousedown', e => { if (e.target === modal) close(); });
  modal.querySelector('.confirm-cancel').onclick = close;
  modal.querySelector('.confirm-ok').onclick = () => { close(); onConfirm(); };
  modal.querySelector('.confirm-ok').focus();
}
async function patchRelation(id, fields) {
  const rel = S.relations.find(r => r.id === id);
  if (rel) Object.assign(rel, fields);
  await api(`/api/relations/${id}`, 'PATCH', fields);
  renderRightPanel();
}
function relDetailHTML(rel) {
  const sr = ruleById(rel.source_id), tr = ruleById(rel.target_id);
  // The Type select edits the USER label. Show the current effective value selected.
  const cur = effectiveType(rel);
  const opts = REL_TYPES.map(t => `<option value="${esc(t)}"${t === cur ? ' selected' : ''}>${esc(t)}</option>`).join('');
  // Contrast: only when an LLM label exists do we surface user-vs-llm.
  let contrast = '';
  if (rel.llm_relation_type) {
    const overridden = rel.relation_type && rel.relation_type !== rel.llm_relation_type;
    contrast = `<div class="rel-contrast">
      <span class="rc-llm">LLM: <b style="color:${relColor(rel.llm_relation_type)}">${esc(rel.llm_relation_type)}</b></span>
      ${rel.relation_type ? `<span class="rc-user">you: <b style="color:${relColor(rel.relation_type)}">${esc(rel.relation_type)}</b></span>` : '<span class="rc-user rc-none">you: (using LLM)</span>'}
      ${overridden ? `<button class="rc-revert" title="Revert to the LLM label" onclick="patchRelation('${rel.id}', {relation_type: null})">revert</button>` : ''}
    </div>`;
  }
  return `<div class="rel-detail" onclick="event.stopPropagation()">
    <div class="rel-full rel-jump" title="Jump to this line in the document" onclick="scrollDocToNode('${rel.source_id}')"><span class="rel-role" style="background:${typeBadgeColor(sr)}">source</span><span>${esc(sr ? sr.rule_text : '(missing)')}</span></div>
    <div class="rel-full rel-jump" title="Jump to this line in the document" onclick="scrollDocToNode('${rel.target_id}')"><span class="rel-role" style="background:${typeBadgeColor(tr)}">target</span><span>${esc(tr ? tr.rule_text : '(missing)')}</span></div>
    <div class="insp-label">Type</div>
    <select class="rel-type-sel" onchange="patchRelation('${rel.id}', {relation_type: this.value})">${opts}</select>
    ${contrast}
    <div class="insp-label">Comment</div>
    <textarea class="insp-comment" placeholder="Note on this relation…"
      onblur="patchRelation('${rel.id}', {notes: this.value})">${esc(rel.notes || '')}</textarea>
    <div class="insp-actions"><span class="insp-status"></span>
      <button class="insp-del" onclick="deleteRelation(event, '${rel.id}')">Delete relation</button></div>
  </div>`;
}
function relFilterHTML() {
  const types = [...new Set(S.relations.map(effectiveType).filter(Boolean))].sort();
  const cnt = t => S.relations.filter(r => effectiveType(r) === t).length;
  const sel = v => S.relFilter === v ? ' selected' : '';
  const opt = (v, label, n) => `<option value="${esc(v)}"${sel(v)}>${esc(label)} (${n})</option>`;
  return `<select class="rl-filter-select" onchange="setRelFilter(this.value)">
    ${opt('all', 'All', S.relations.length)}
    ${types.map(t => opt(t, t, cnt(t))).join('')}
  </select>`;
}
function setRelFilter(v) {
  if (v === S.relFilter) return;
  S.relFilter = v;
  if (S.focusedRelId && !visibleRelations().some(r => r.id === S.focusedRelId)) S.focusedRelId = null;
  renderViewer(); renderRightPanel();
}
function visibleRelations() {
  if (S.relFilter === 'all') return S.relations;
  return S.relations.filter(r => effectiveType(r) === S.relFilter);
}
function renderRelationList() {
  const el = document.getElementById('inspector');
  if (!S.currentFile) { el.innerHTML = `<div class="insp-empty">Select a file.</div>`; return; }
  const rels = S.relations;
  if (S.relFilter !== 'all' && !rels.some(r => effectiveType(r) === S.relFilter)) S.relFilter = 'all';
  const head = `<div class="rl-headbar">
    <div class="rl-head">${rels.length} relation${rels.length === 1 ? '' : 's'}</div>
    <div class="rel-head-actions">
      ${rels.length ? relFilterHTML() : ''}
      ${rels.length ? `<button class="rel-delall" onclick="deleteAllRelations()" title="Delete every relation in this file">Delete all</button>` : ''}
    </div>
  </div>`;
  if (!rels.length) { el.innerHTML = head + `<div class="insp-empty">No relations yet.</div>`; return; }
  const vis = visibleRelations();
  if (!vis.length) { el.innerHTML = head + `<div class="insp-empty">No relations match this filter.</div>`; return; }
  const items = vis.map(rel => {
    const focused = rel.id === S.focusedRelId;
    const eff = effectiveType(rel);
    const c = relColor(eff);
    const llm = (rel.source === 'llm' || rel.source === 'revise');
    const overridden = rel.llm_relation_type && rel.relation_type && rel.relation_type !== rel.llm_relation_type;
    return `<div class="rel-item${focused ? ' active' : ''}" id="rel-${rel.id}" onclick="focusRelation('${rel.id}')">
      <div class="rel-row1">
        <span class="rel-type" style="color:${c}">${esc(eff || '—')}</span>
        ${relRingHTML(rel)}
        ${overridden ? `<span class="rel-over" title="You changed the LLM label from ${esc(rel.llm_relation_type)}">✎ edited</span>` : ''}
        ${rel.notes ? '<span class="rl-flag" title="Has comment">✎</span>' : ''}
        ${llm ? '<span class="rel-llm">llm</span>' : ''}
        ${rel.vote_count != null ? `<span class="rl-ring-cnt">${rel.vote_count}/${rel.judge_total}</span>` : ''}
        <span class="rel-caret">${focused ? '▾' : '▸'}</span>
      </div>
      <div class="rel-pair"><span class="rel-node" style="color:${typeBadgeColor(ruleById(rel.source_id))}">${esc(ruleShort(rel.source_id, 38))}</span>
        <button class="rel-arrow" title="Swap source ↔ target" onclick="swapRelation(event, '${rel.id}')">→</button>
        <span class="rel-node" style="color:${typeBadgeColor(ruleById(rel.target_id))}">${esc(ruleShort(rel.target_id, 38))}</span></div>
      ${focused ? relDetailHTML(rel) : ''}
    </div>`;
  }).join('');
  el.innerHTML = head + items;
}

// collapse the currently expanded rule (Esc, red ✕)
function exitInspector() { setFocusedRule(null); }

// accordion: clicking a rule expands it; clicking it again (or another) collapses
function toggleRuleExpand(id, evt) {
  evt?.stopPropagation();
  setFocusedRule(S.focusedRuleId === id ? null : id);
}

function renderRuleList() {
  const el = document.getElementById('inspector');
  if (!S.currentFile) {
    el.innerHTML = `<div class="insp-empty">Select a file, then click a highlight or select text and press <b>+ Add</b>.</div>`;
    return;
  }
  if (!S.rules.length) {
    const hint = S.mode === 'label'
      ? `No rules to label yet.<br><br>Switch to <b>Extract</b> to identify rules first, then open <b>LLM judge</b> here to vote tags.`
      : `No rules yet.<br><br>Open <b>LLM judge</b> and press <b>Run</b> to let the judges extract rules, or select text and press <b>+ Add</b> to mark one yourself.`;
    el.innerHTML = `<div class="insp-empty">${hint}</div>`;
    return;
  }
  const vis = visibleRules();          // source-filtered (the source filter hides; type/search fade within)
  const active = filterActive();
  const matchN = active ? vis.filter(matchesFilter).length : vis.length;
  const noun = S.mode === 'label' ? 'rule' : 'item';
  const head = `<div class="rl-headbar">
    <div class="rl-head-top">
      <div class="rl-head">${vis.length} ${noun}${vis.length === 1 ? '' : 's'}${active ? ` · ${matchN} match` : ''}</div>
      <input type="search" class="rl-search" id="ruleSearch" placeholder="Search keyword…"
        value="${escAttr(S.search || '')}" oninput="liveSearch(this.value)" autocomplete="off" spellcheck="false">
    </div>
    ${filterBarHTML()}
  </div>`;
  const items = vis.map((r) => {
    const col = listBarColor(r);
    const expanded = r.id === S.focusedRuleId;
    const faded = active && !matchesFilter(r);
    const tag = badgeHTML(r);
    const flags = [
      (S.mode === 'label' && tagRevised(r)) ? `<span class="rl-flag rl-flag-rev" title="You re-tagged this (LLM picked ${esc(r.llm_tag)})">↺</span>` : '',
      r.llm_rationale ? '<span class="rl-flag" title="Has LLM rationale">⚖</span>' : '',
      r.notes ? '<span class="rl-flag" title="Has comment">✎</span>' : '',
    ].join('');
    const preview = r.rule_text.replace(/\s+/g, ' ').slice(0, 90);
    return `<div class="rl-item${expanded ? ' active expanded' : ''}${faded ? ' rl-faded' : ''}" id="rl-${r.id}" style="${rcVars(col)}">
      <div class="rl-bar"></div>
      <div class="rl-body">
        <div class="rl-header" onclick="toggleRuleExpand('${r.id}', event)">
          <div class="rl-top">${tag}<span class="rl-pos">${posLabel(r)}</span>
            <span class="rl-flags">${flags}<span class="rl-caret">${expanded ? '▾' : '▸'}</span></span></div>
          <div class="rl-text">${esc(preview)}${r.rule_text.length > 90 ? '…' : ''}</div>
        </div>
        ${expanded ? ruleDetailHTML(r) : ''}
      </div>
    </div>`;
  }).join('');
  el.innerHTML = head + items;
  if (S.focusedRuleId) {
    autoGrow(document.getElementById('inspRationale'));
    autoGrow(document.getElementById('inspComment'));
  }
  renderKbBar();   // reflect whether the focused rule is awaiting a tag
}

// Two dropdowns: SOURCE (human/LLM) HIDES to show only that category; TYPE (deontic
// tags · context sub-types) FADES non-matches. Counts are cross-aware (each respects
// the other dropdown's selection).
function filterBarHTML() {
  const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
  const cntSrc  = src => S.rules.filter(r => matchesType(r) && matchesSrc(r, src)).length;
  const cntType = ft  => S.rules.filter(r => matchesType(r, ft) && matchesSrc(r)).length;
  const selS = v => curFilterSrc() === v ? ' selected' : '';
  const selT = v => curFilterType() === v ? ' selected' : '';
  const optS = (v, label) => `<option value="${v}"${selS(v)}>${label} (${cntSrc(v)})</option>`;
  const optT = (v, label) => `<option value="${v}"${selT(v)}>${label} (${cntType(v)})</option>`;
  // Extract = provenance (you / LLM). Label = show only rules a human re-tagged.
  const src = (S.mode === 'label')
    ? `<select class="rl-filter-select" title="Show only rules you re-tagged (the LLM's pick is kept)" onchange="setFilterSrc(this.value)">
        ${optS('all', 'All')}${optS('revised', 'Revised')}
      </select>`
    : `<select class="rl-filter-select" title="Show only this category (you / LLM)" onchange="setFilterSrc(this.value)">
        ${optS('all', 'All')}${optS('hand', 'You')}${optS('llm', 'LLM')}
      </select>`;
  // Extract = filter by KIND (Rule/Context, the green/yellow axis). Label = by
  // deontic tag (+ Untagged) and context sub-type.
  const type = (S.mode === 'extract')
    ? `<select class="rl-filter-select" title="Filter by kind — fades non-matches" onchange="setFilterType(this.value)">
        ${optT('all', 'All kinds')}${optT('rule', 'Rule')}${optT('context', 'Context')}
      </select>`
    : `<select class="rl-filter-select" title="Filter by tag / context type — fades non-matches" onchange="setFilterType(this.value)">
        ${optT('all', 'All types')}
        <optgroup label="Rule">
          ${optT('rule', 'Rule · All')}
          ${TAGS.map(t => optT('rule:' + t, cap(t.toLowerCase()))).join('')}
          ${optT('rule:none', 'Untagged')}
        </optgroup>
        <optgroup label="Context">
          ${optT('context', 'Context · All')}
          ${CONTEXT_TYPES.map(t => optT('context:' + t, cap(t))).join('')}
        </optgroup>
      </select>`;
  return `<div class="rl-filter-row">${src}${type}</div>`;
}

function moveFocus(delta) {
  const vis = visibleRules();
  if (!vis.length) return;
  const idx = vis.findIndex(r => r.id === S.focusedRuleId);
  const next = idx < 0
    ? (delta > 0 ? 0 : vis.length - 1)
    : clamp(idx + delta, 0, vis.length - 1);
  setFocusedRule(vis[next].id);
}

// ─── Inspector ───────────────────────────────────────────────
function posLabel(r) {
  if (r.line_start) return `Line ${r.line_start}${r.line_end && r.line_end !== r.line_start ? '–' + r.line_end : ''}`;
  if (r.char_start != null) return `chars ${r.char_start}–${r.char_end}`;
  return 'no position';
}

// Expanded detail rendered inline inside the focused rule's accordion card.
function ruleDetailHTML(r) {
  // Extract mode = decide the SPAN's kind (Rule/Context). Label mode = assign the
  // deontic tag (or context sub-type) and inspect the judge votes.
  const kindToggle = (S.mode === 'extract') ? `<div class="kind-toggle k-${ruleKind(r)}">
      <button class="${ruleKind(r) === 'context' ? 'active' : ''}" onclick="setKind('context')">Context</button>
      <button class="${ruleKind(r) === 'rule' ? 'active' : ''}" onclick="setKind('rule')">Rule</button>
    </div>` : '';
  const tagSec = (S.mode === 'label') ? `<div id="tagSection">${tagSectionHTML(r)}</div>` : '';
  const ring = (S.mode === 'label') ? voteInspectRow(r) : '';
  const ratio = (r.source === 'hand') ? '' : `<div class="insp-label">LLM Rationale</div>
    <textarea class="insp-rationale" id="inspRationale" readonly placeholder="Run the LLM judge (top bar) to populate this…">${esc(r.llm_rationale || '')}</textarea>`;
  return `<div class="rl-detail">
    ${kindToggle}
    ${tagSec}
    ${ring}
    ${ratio}
    <div class="insp-label">Comment</div>
    <textarea class="insp-comment" id="inspComment" placeholder="Your comment… (Enter to save · Shift+Enter for newline)"
      oninput="autoGrow(this)" onblur="saveComment()" onkeydown="commentKey(event)">${esc(r.notes || '')}</textarea>

    <div class="insp-actions">
      <span class="insp-status" id="inspStatus"></span>
      <button class="insp-del" onclick="deleteFocusedRule()">Delete ${ruleKind(r) === 'context' ? 'context' : 'rule'}</button>
    </div>
  </div>`;
}
// A "judge votes: 2/3 PRESCRIPTION — inspect" row in the Label detail panel.
function voteInspectRow(r) {
  const info = S.judge.label;
  const votes = info && info.byRule ? info.byRule[r.id] : null;
  if (!votes || !votes.length) return '';
  const ring = voteRingHTML(votes, 'label',
    `event.stopPropagation();openVoteInspector('label','${r.id}')`, 'inspect all judges');
  return `<div class="insp-label">Judge votes</div>
    <div class="rl-ring-wrap" style="cursor:pointer" onclick="openVoteInspector('label','${r.id}')">
      ${ring}<span class="rl-ring-cnt">${votes.length} judges · click to inspect</span></div>`;
}

// Type picker: deontic tag for 'rule' items, sub-type for 'context' items.
function tagSectionHTML(r) {
  if (ruleKind(r) === 'context') {
    const btns = CONTEXT_TYPES.map(t =>
      `<button class="tag-btn ctx${r.context_type === t ? ' active ctx' : ''}" onclick="setContextType('${t}')">${t[0].toUpperCase() + t.slice(1)}</button>`
    ).join('');
    return `<div class="insp-label">Context type</div><div class="tag-row ctx">${btns}</div>`;
  }
  const tagBtns = TAGS.map(t =>
    `<button class="tag-btn${r.tag === t ? ' active ' + t.toLowerCase() : ''}" onclick="setTag('${t}')">${t}</button>`
  ).join('');
  // Show the LLM vote's pick so a human override is visible and revertable (its choice
  // is preserved in llm_tag). Re-rendered in place by refreshTagSection on every change.
  let contrast = '';
  if (r.llm_tag) {
    contrast = tagRevised(r)
      ? `<div class="tag-contrast revised">LLM picked <b style="color:${TYPE_BADGE[r.llm_tag] || '#888'}">${esc(r.llm_tag)}</b> — you changed it
           <button class="tag-revert" onclick="revertTagToLLM()" title="Restore the LLM's tag">↺ revert</button></div>`
      : `<div class="tag-contrast">LLM picked <b style="color:${TYPE_BADGE[r.llm_tag] || '#888'}">${esc(r.llm_tag)}</b> — kept</div>`;
  }
  return `<div class="insp-label">Tag</div><div class="tag-row">${tagBtns}</div>${contrast}`;
}
// Restore the LLM vote's tag on the focused rule (clears the "revised" state).
async function revertTagToLLM() {
  const r = ruleById(S.focusedRuleId);
  if (!r || !r.llm_tag || r.tag === r.llm_tag) return;
  r.tag = r.llm_tag;
  refreshTagSection(r);     // updates buttons + contrast + header badge in place
  refreshCardColor(r);
  renderKbBar();
  await api(`/api/rules/${r.id}`, 'PATCH', { tag: r.tag });
  // 'Revised' filter membership may have changed — keep the list count honest.
  if (curFilterSrc() === 'revised') renderRuleList();
}

async function setKind(k) {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r || ruleKind(r) === k) return;
  r.kind = k;
  const patch = { kind: k };
  if (k === 'context') { r.tag = null; patch.tag = null; }            // rules-only deontic tag
  else { r.context_type = null; patch.context_type = null; }          // context-only sub-type
  if (markReviewed(r)) patch.reviewed = true;                          // changing kind = human review
  await api(`/api/rules/${r.id}`, 'PATCH', patch);
  renderViewer();        // highlight colour changes (grey for context); re-paints purple
  renderRightPanel();    // re-render: swap Tag↔Context-type picker, update filter membership
}

// Auto-grow a textarea to fit its content, up to its CSS max-height, then scroll.
function autoGrow(el) {
  if (!el) return;
  el.style.height = 'auto';
  const max = parseInt(getComputedStyle(el).maxHeight, 10) || 320;
  const h = Math.min(el.scrollHeight, max);
  el.style.height = h + 'px';
  el.style.overflowY = el.scrollHeight > max ? 'auto' : 'hidden';
}

function setInspStatus(msg, cls = '') {
  const el = document.getElementById('inspStatus');
  if (el) { el.textContent = msg; el.className = 'insp-status' + (cls ? ' ' + cls : ''); }
}

// Solid colour for a vote value, per mode (Label = deontic tag; Relation = edge type).
function voteColor(mode, val) {
  if (mode === 'relation') return relColor(val);
  return TYPE_BADGE[val] || '#b8b8c4';   // label: tag colour; null/abstain → grey
}
// Conic-gradient agreement ring from a list of votes. `onclick` opens the inspector.
function voteRingHTML(votes, mode, onclick, title) {
  if (!votes || !votes.length) return '';
  const n = votes.length, seg = 100 / n;
  const stops = votes.map((v, i) =>
    `${voteColor(mode, v.value)} ${(i * seg).toFixed(2)}% ${((i + 1) * seg).toFixed(2)}%`).join(', ');
  return `<span class="vote-ring" title="${escAttr(title || '')}" onclick="${onclick}"`
    + ` style="background:conic-gradient(${stops})"></span>`;
}
// Centered modal: every judge's vote + short rationale for one rule (Label) or edge (Relation).
function openVoteInspector(mode, key) {
  const info = S.judge[mode];
  if (!info) return;
  let votes, quote;
  if (mode === 'relation') {
    votes = (info.byEdge || {})[key] || [];
    const [s, t] = key.split('::');
    quote = `${ruleShort(s, 56)}  →  ${ruleShort(t, 56)}`;
  } else {
    votes = (info.byRule || {})[key] || [];
    const r = ruleById(key); quote = r ? r.rule_text : '';
  }
  const tally = {};
  for (const v of votes) { const k = v.value || '—'; tally[k] = (tally[k] || 0) + 1; }
  const total = votes.length || 1;
  const chips = Object.entries(tally).sort((a, b) => b[1] - a[1]).map(([val, n]) =>
    `<span class="vi-chip" style="background:${voteColor(mode, val === '—' ? null : val)}">`
    + `${esc(val)} ${n}/${votes.length} · ${Math.round(100 * n / total)}%</span>`).join('');
  const rows = votes.map(v => `<tr>
    <td class="vi-model">${esc((v.model || '').split('/').pop())}</td>
    <td class="vi-vote" style="color:${voteColor(mode, v.value)}">${esc(v.value || '—')}</td>
    <td class="vi-rat">${esc(v.rationale || '')}</td></tr>`).join('');
  document.getElementById('voteInspector')?.remove();
  const modal = document.createElement('div');
  modal.id = 'voteInspector'; modal.className = 'modal-backdrop';
  modal.addEventListener('mousedown', e => { if (e.target === modal) modal.remove(); });
  modal.innerHTML = `<div class="modal-card vi-card">
    <div class="modal-head"><span class="tool-title">Judge votes · ${esc(mode)}</span>
      <span style="font-size:11px;color:#9090b0;margin-left:6px">${info.judge_total} judges · accept ≥ ${info.threshold}</span>
      <div style="flex:1"></div>
      <button class="insp-exit" onclick="document.getElementById('voteInspector').remove()" title="Close (Esc)">✕</button></div>
    <div class="modal-body" style="overflow:auto;display:block">
      <div class="vi-quote">${esc(quote)}</div>
      <div class="vi-tally">${chips || '<span style="color:#aaa;font-size:12px">no votes recorded</span>'}</div>
      <table class="vi-table"><thead><tr><th>Judge</th><th>Vote</th><th>Rationale</th></tr></thead>
      <tbody>${rows}</tbody></table>
    </div></div>`;
  document.body.appendChild(modal);
}

// Header badge: Extract → the span's kind (RULE/CONTEXT). Label → deontic tag /
// context sub-type, plus the vote ring (click to inspect).
function kindBadge(r) {
  const k = ruleKind(r);
  return `<span class="rl-kind ${k}">${k.toUpperCase()}</span>`;
}
function tagBadgeOnly(r) {
  if (ruleKind(r) === 'context')
    return `<span class="rl-tag context">${r.context_type ? r.context_type.toUpperCase() : 'CONTEXT'}</span>`;
  return r.tag ? `<span class="rl-tag ${r.tag.toLowerCase()}">${r.tag}</span>`
              : `<span class="rl-tag" style="background:#eceef2;color:#8a8aa6">UNTAGGED</span>`;
}
function labelRingHTML(r) {
  const info = S.judge.label;
  const votes = info && info.byRule ? info.byRule[r.id] : null;
  if (!votes || !votes.length) return '';
  const ring = voteRingHTML(votes, 'label',
    `event.stopPropagation();openVoteInspector('label','${r.id}')`,
    votes.length + ' judge votes — click to inspect');
  return `<span class="rl-ring-wrap">${ring}</span>`;
}
function relRingHTML(rel) {
  const info = S.judge.relation;
  const key = rel.source_id + '::' + rel.target_id;
  const votes = info && info.byEdge ? info.byEdge[key] : null;
  if (!votes || !votes.length) return '';
  return voteRingHTML(votes, 'relation',
    `event.stopPropagation();openVoteInspector('relation','${key}')`,
    votes.length + ' judge votes — click to inspect');
}
function badgeHTML(r) {
  if (S.mode === 'extract') return kindBadge(r);
  return tagBadgeOnly(r) + labelRingHTML(r);
}

function refreshTagSection(r) {
  const el = document.getElementById('tagSection');
  if (el) el.innerHTML = tagSectionHTML(r);
  // keep the collapsed-header tag badge in sync (in place — preserves textareas + ring)
  const top = document.getElementById('rl-' + r.id)?.querySelector('.rl-top');
  if (top) {
    const html = tagBadgeOnly(r);
    const old = top.querySelector('.rl-tag');
    if (old) old.outerHTML = html;
    else top.insertAdjacentHTML('afterbegin', html);
  }
}

// Mark an LLM item as human-reviewed (purple). Returns true if it just flipped.
function markReviewed(r) {
  if (r.source === 'hand' || r.reviewed) return false;
  r.reviewed = true;
  return true;
}

// toggle=true (button click): clicking the active tag clears it. toggle=false
// (a/s/d/f keyboard): always SET the tag, so re-pressing the current key is a no-op.
async function setTag(t, toggle = true) {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r) return;
  const newTag = (toggle && r.tag === t) ? null : t;
  if (newTag === r.tag) return;             // no change (e.g. keyboard 'set' on the current tag)
  r.tag = newTag;
  const patch = { tag: newTag };
  const flipped = markReviewed(r);          // editing an LLM tag = human review
  if (flipped) patch.reviewed = true;
  refreshTagSection(r);                     // update buttons + LLM contrast in place
  if (flipped) refreshCardColor(r);         // blue → purple in place
  renderKbBar();                            // tag set/cleared → refresh the a/s/d/f hint
  // The "Revised" filter membership can change with the tag — re-render the list so a
  // rule appears/disappears from the filtered view (and the count stays honest).
  if (curFilterSrc() === 'revised') renderRuleList();
  await api(`/api/rules/${r.id}`, 'PATCH', patch);
}

// Context sub-type (condition / reference / definition) — parallel to setTag.
async function setContextType(t) {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r) return;
  r.context_type = r.context_type === t ? null : t;   // click the active one to clear
  const patch = { context_type: r.context_type };
  const flipped = markReviewed(r);
  if (flipped) patch.reviewed = true;
  refreshTagSection(r);                                // update buttons + header badge in place
  if (flipped) refreshCardColor(r);
  await api(`/api/rules/${r.id}`, 'PATCH', patch);
}

async function saveComment() {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  const inp = document.getElementById('inspComment');
  if (!r || !inp) return;
  const notes = inp.value.trim() || null;
  if (notes === (r.notes || null)) return;
  const wasReviewed = isReviewed(r);
  r.notes = notes;
  if (isReviewed(r) !== wasReviewed) refreshCardColor(r);   // comment added/removed → toggle purple
  await api(`/api/rules/${r.id}`, 'PATCH', { notes });
  setInspStatus('Comment saved', 'ok');
}

function commentKey(e) {
  // Enter saves & finishes; Shift+Enter inserts a newline.
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.target.blur(); }
  e.stopPropagation();
}

async function deleteFocusedRule() {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r) return;
  await api(`/api/rules/${r.id}`, 'DELETE');
  S.rules = S.rules.filter(x => x.id !== r.id);
  S.focusedRuleId = null;          // collapse to the list — don't auto-open the next rule
  renderViewer();
  renderRuleList();
  refreshFileBadge(S.currentFile?.id);
}


// ─── Tool view (editable prompts) ────────────────────────────
const MODEL_OPTIONS = `
  <optgroup label="Perplexity">
    <option value="sonar">sonar</option>
    <option value="sonar-pro">sonar-pro</option>
    <option value="sonar-reasoning-pro">sonar-reasoning-pro</option>
    <option value="sonar-deep-research">sonar-deep-research</option>
  </optgroup>
  <optgroup label="Anthropic (via Perplexity)">
    <option value="anthropic/claude-haiku-4-5">claude-haiku-4-5</option>
    <option value="anthropic/claude-sonnet-4-5">claude-sonnet-4-5</option>
    <option value="anthropic/claude-sonnet-4-6">claude-sonnet-4-6</option>
    <option value="anthropic/claude-opus-4-5">claude-opus-4-5</option>
    <option value="anthropic/claude-opus-4-6">claude-opus-4-6</option>
    <option value="anthropic/claude-opus-4-7">claude-opus-4-7</option>
    <option value="anthropic/claude-opus-4-8">claude-opus-4-8</option>
  </optgroup>
  <optgroup label="OpenAI (via Perplexity)">
    <option value="openai/gpt-5-mini">gpt-5-mini</option>
    <option value="openai/gpt-5">gpt-5</option>
    <option value="openai/gpt-5.1">gpt-5.1</option>
    <option value="openai/gpt-5.4-mini">gpt-5.4-mini</option>
    <option value="openai/gpt-5.4">gpt-5.4</option>
    <option value="openai/gpt-5.5">gpt-5.5</option>
  </optgroup>`;

function toggleTool() {
  if (!S.currentFile || S.judgeRunning) return;   // locked while a run is in flight
  S.toolMode = !S.toolMode;
  document.getElementById('judgeBtn').classList.toggle('active', S.toolMode);
  renderJudgeModal();
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// The loading screen: one progress bar PER judge so you can watch them run in
// parallel and finish independently (indeterminate while running → solid on done).
function judgeProgressHTML() {
  const p = S.judgeProgress;
  if (!p) return `<div class="spinner"></div><div class="judge-loading-text">Starting judges…</div>`;
  const doneN = p.judges.filter(j => j.status !== 'running').length;
  // All judges returned → the server is now clustering votes + recording the accepted
  // results. Tell the user so the finished bars don't look stuck.
  const aggregating = doneN === p.judge_total;
  const noun = p.mode === 'relation' ? 'relations' : (p.mode === 'label' ? 'tags' : 'rules');
  const rows = p.judges.map((j, i) => `
    <div class="jp-item">
      <span class="jp-name">Judge ${i + 1} · ${esc((j.model || '').split('/').pop())}</span>
      <div class="jp-bar jp-${j.status}"><div class="jp-fill"></div></div>
    </div>`).join('');
  const head = aggregating
    ? `Judges done — aggregating votes & recording accepted ${noun}…`
    : `Running ${p.judge_total} judge${p.judge_total === 1 ? '' : 's'} in parallel · ${esc(modeDef(p.mode).label)} — ${doneN}/${p.judge_total} done`;
  const aggRow = aggregating
    ? `<div class="jp-item jp-agg"><span class="jp-name">Aggregating accepted ${noun}…</span>
         <div class="jp-bar jp-running"><div class="jp-fill"></div></div></div>`
    : '';
  return `<div class="judge-loading-text">${head}</div>
    <div class="jp-list">${rows}${aggRow}</div>`;
}
// Update the loading bars in place (no full modal rebuild → no flicker).
function renderJudgeProgress() {
  const el = document.getElementById('judgeLoading');
  if (el) el.innerHTML = judgeProgressHTML();
}

// The LLM judge is a centered pop-up modal: a panel of judges (one model each), an
// acceptance threshold, and the prompt — for the CURRENT pipeline stage (S.mode).
function renderJudgeModal() {
  let modal = document.getElementById('judgeModal');
  if (!S.toolMode) { if (modal) modal.remove(); return; }
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'judgeModal';
    modal.className = 'modal-backdrop';
    modal.addEventListener('mousedown', e => { if (e.target === modal && !S.judgeRunning) toggleTool(); });
    document.body.appendChild(modal);
  }
  if (S.judgeRunning) {
    modal.innerHTML = `<div class="modal-card"><div class="judge-loading" id="judgeLoading">${judgeProgressHTML()}</div></div>`;
    return;
  }
  const md = modeDef(S.mode);
  const cfg = S.judgeCfg[S.mode];
  const N = cfg.judges.length;
  const judgeRows = cfg.judges.map((j, i) => `
    <div class="judge-item">
      <span class="jn">Judge ${i + 1}</span>
      <select onchange="setJudgeModel(${i}, this.value)">${MODEL_OPTIONS}</select>
      ${N > 1 ? `<button class="judge-del" title="Remove judge" onclick="removeJudge(${i})">✕</button>` : ''}
    </div>`).join('');
  // Stage gating.
  const rules = S.rules.filter(r => (r.kind || 'rule') === 'rule');
  const untaggedN = rules.filter(r => !r.tag).length;
  let gate = '', disabled = false;
  if (S.mode === 'label' && rules.length < 1) { gate = 'No rules to label yet — run Extract first.'; disabled = true; }
  if (S.mode === 'relation') {
    if (S.rules.length < 2) { gate = 'Need ≥ 2 rules/context to relate — run Extract first.'; disabled = true; }
    else if (untaggedN > 0) { gate = `Tag every rule in Label first — ${untaggedN} still untagged. Relations are built from context + tagged rules.`; disabled = true; }
  }
  const tabs = MODE_DEFS.map(m =>
    `<button class="jm-tab${m.id === S.mode ? ' active' : ''}" onclick="judgeSwitchMode('${m.id}')">${m.label}</button>`).join('');
  modal.innerHTML = `
    <div class="modal-card">
      <div class="modal-head">
        <span class="tool-title">LLM judge</span>
        <div class="judge-modes">${tabs}</div>
        <div style="flex:1"></div>
        <button class="insp-exit" onclick="toggleTool()" title="Back to document (Esc)">✕</button>
      </div>
      <div class="modal-body">
        <div class="judge-cfg">
          <div class="judge-cfg-row"><span class="lbl">Judges</span><span class="tool-status" style="flex:1">Each judge votes independently; ${md.label} is decided by their agreement.</span></div>
          <div class="judge-list">${judgeRows}</div>
          <button class="judge-add" onclick="addJudge()"${N >= 7 ? ' disabled' : ''}>+ Add judge</button>
          <div class="thr-row">
            <span class="lbl">Accept ≥</span>
            <input type="range" min="1" max="${N}" value="${cfg.threshold}" oninput="setThreshold(this.value)">
            <span class="thr-val" id="thrVal">${cfg.threshold} / ${N} judges <small>(${Math.round(100 * cfg.threshold / N)}%)</small></span>
          </div>
        </div>
        <textarea class="tool-prompt" id="judgePromptArea" oninput="S.judgeCfg['${S.mode}'].prompt=this.value">${esc(cfg.prompt || '')}</textarea>
      </div>
      <div class="modal-foot">
        <span class="tool-status${disabled ? ' error' : ''}" id="judgeToolStatus">${gate || md.blurb}</span>
        <button class="btn ghost" onclick="restoreDefaultPrompt()" title="Reset this prompt to the built-in default">↺ Restore default</button>
        <button class="btn primary" id="judgeRunBtn" onclick="runJudge()"${disabled ? ' disabled' : ''}>▶ Run ${md.label} · ${N} judge${N === 1 ? '' : 's'}</button>
      </div>
    </div>`;
  // restore each judge's selected model
  modal.querySelectorAll('.judge-item select').forEach((sel, i) => { if (cfg.judges[i]) sel.value = cfg.judges[i].model; });
}

// Switch pipeline stage from inside the modal (also flips the interface mode).
function judgeSwitchMode(m) {
  if (m === S.mode) return;
  const area = document.getElementById('judgePromptArea');
  if (area) S.judgeCfg[S.mode].prompt = area.value;   // stash in-progress edit
  setMode(m);
  renderJudgeModal();
}
function setJudgeModel(i, v) { const c = S.judgeCfg[S.mode]; if (c.judges[i]) c.judges[i].model = v; }
function addJudge() {
  const c = S.judgeCfg[S.mode];
  if (c.judges.length >= 7) return;
  c.judges.push({ model: DEFAULT_JUDGE_MODELS[c.judges.length % DEFAULT_JUDGE_MODELS.length] || 'sonar' });
  renderJudgeModal();
}
function removeJudge(i) {
  const c = S.judgeCfg[S.mode];
  if (c.judges.length <= 1) return;
  c.judges.splice(i, 1);
  if (c.threshold > c.judges.length) c.threshold = c.judges.length;
  renderJudgeModal();
}
function setThreshold(v) {
  const c = S.judgeCfg[S.mode];
  c.threshold = Math.max(1, Math.min(parseInt(v, 10) || 1, c.judges.length));
  const el = document.getElementById('thrVal');
  if (el) el.innerHTML = `${c.threshold} / ${c.judges.length} judges <small>(${Math.round(100 * c.threshold / c.judges.length)}%)</small>`;
}

function setToolStatus(id, msg, cls = '') {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.className = 'tool-status' + (cls ? ' ' + cls : ''); }
}


// ─── File comment (⌘?) ───────────────────────────────────────
// A per-(file, annotator) free-text comment, in a pop-up panel parallel to the
// LLM judge. Saved to the DB (file_comments) and reloaded with the file.
function toggleComment() {
  if (!S.currentFile || S.judgeRunning || S.toolMode) return;  // not on top of the judge
  S.commentMode = !S.commentMode;
  document.getElementById('commentBtn').classList.toggle('active', S.commentMode);
  renderCommentModal();
}

function renderCommentModal() {
  let modal = document.getElementById('commentModal');
  if (!S.commentMode) { if (modal) modal.remove(); return; }
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'commentModal';
    modal.className = 'modal-backdrop';
    // click on the dimmed backdrop (outside the card) closes it
    modal.addEventListener('mousedown', e => { if (e.target === modal) toggleComment(); });
    document.body.appendChild(modal);
  }
  modal.innerHTML = `
    <div class="modal-card" style="max-width:560px">
      <div class="modal-head">
        <span class="tool-title">File comment</span>
        <span style="font-size:11px;color:#9090b0;margin-left:8px">${esc(S.userName || 'no annotator selected')}</span>
        <div style="flex:1"></div>
        <button class="insp-exit" onclick="toggleComment()" title="Back to document (Esc)">✕</button>
      </div>
      <div class="modal-body">
        <textarea class="tool-prompt" id="fileCommentArea" style="min-height:140px"
          placeholder="Your comment on this file… (visible only under your annotator name)"
          onkeydown="if((event.metaKey||event.ctrlKey)&&event.key==='Enter'){event.preventDefault();saveFileComment();}"
        >${esc(S.fileComment || '')}</textarea>
      </div>
      <div class="modal-foot">
        <span class="tool-status" id="commentToolStatus">Sticks to this file for annotator “${esc(S.userName || '?')}”. ⌘↵ to save.</span>
        <button class="btn primary" id="commentSaveBtn" onclick="saveFileComment()">Save</button>
      </div>
    </div>`;
  const ta = document.getElementById('fileCommentArea');
  ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
}

async function loadFileComment() {
  if (!S.currentFile || !S.userName) { S.fileComment = ''; updateCommentBtn(); return; }
  const r = await api(`/api/file-comment/${S.currentFile.id}?annotator=${encodeURIComponent(S.userName)}`);
  S.fileComment = (r && typeof r.comment === 'string') ? r.comment : '';
  updateCommentBtn();
}

async function saveFileComment() {
  const ta = document.getElementById('fileCommentArea');
  if (!ta || !S.currentFile) return;
  if (!S.userName) { setToolStatus('commentToolStatus', 'Pick an annotator first (top right)', 'err'); return; }
  const txt = ta.value.trim();
  const r = await api(`/api/file-comment/${S.currentFile.id}`, 'POST', { annotator: S.userName, comment: txt });
  if (r && r.ok) {
    S.fileComment = txt;
    updateCommentBtn();
    setToolStatus('commentToolStatus', txt ? 'Saved ✓' : 'Comment cleared ✓', 'ok');
  } else {
    setToolStatus('commentToolStatus', (r && r.error) || 'Save failed', 'err');
  }
}

// green-filled button when the open file already carries a comment from this annotator
function updateCommentBtn() {
  const btn = document.getElementById('commentBtn');
  if (btn) btn.classList.toggle('has-comment', !!S.fileComment);
}

// Prompts are saved per-annotator; judge_model stays global. The annotator is
// attached here so every caller persists under the right person.
async function persistSettings(obj) { await api('/api/settings', 'POST', { ...obj, annotator: S.userName || '' }); }

// "Restore default": reset the current stage's prompt to the built-in default and
// drop this annotator's saved config override.
async function restoreDefaultPrompt() {
  const md = modeDef(S.mode);
  S.judgeCfg[S.mode].prompt = md.deflt;
  await api('/api/settings/' + md.key + '?annotator=' + encodeURIComponent(S.userName || ''), 'DELETE');
  renderJudgeModal();
  setToolStatus('judgeToolStatus', 'Restored the default prompt for this annotator.', 'ok');
}

// Run the multi-judge pass for the current stage. Kicks off a background job and
// polls its progress so the loading screen shows each judge finishing in parallel,
// then reloads rules / relations / votes.
async function runJudge() {
  if (!S.currentFile || S.judgeRunning) return;
  const mode = S.mode;
  const cfg = S.judgeCfg[mode];
  const area = document.getElementById('judgePromptArea');
  if (area) cfg.prompt = area.value;
  const prompt = (cfg.prompt || '').trim();
  if (!prompt) { setToolStatus('judgeToolStatus', 'Enter a prompt', 'error'); return; }

  S.judgeRunning = true;
  S.judgeProgress = { mode, judge_total: cfg.judges.length,
                      judges: cfg.judges.map(j => ({ model: j.model, status: 'running' })) };
  renderJudgeModal();   // → per-judge progress bars
  // Persist this stage's whole config (judges + threshold + prompt) per annotator.
  await persistSettings({ [modeDef(mode).key]: JSON.stringify({ threshold: cfg.threshold, judges: cfg.judges, prompt: cfg.prompt }) });

  const started = await api('/api/judge', 'POST', {
    file_id: S.currentFile.id, mode, prompt, threshold: cfg.threshold,
    judges: cfg.judges, annotator: S.userName || 'unknown',
  });
  if (started.error || !started.run_id) {
    S.judgeRunning = false; S.judgeProgress = null;
    renderJudgeModal(); setToolStatus('judgeToolStatus', started.error || 'Failed to start', 'error');
    return;
  }

  // Poll per-judge progress until the job finishes.
  let prog = null;
  for (let i = 0; i < 900; i++) {            // ~10 min ceiling at 700ms
    await sleep(700);
    prog = await api('/api/judge-progress/' + started.run_id);
    if (prog && prog.missing) break;         // job expired/lost
    if (prog && Array.isArray(prog.judges)) {
      S.judgeProgress.judges = prog.judges;
      renderJudgeProgress();
    }
    if (prog && prog.done) break;
  }

  S.judgeRunning = false; S.judgeProgress = null;

  // Reload latest data REGARDLESS of how the poll ended — a timed-out or lost-contact
  // job may still have committed, and a transient blip on the final poll must not be
  // mistaken for failure (the server sets `error` only together with `done`).
  const res = (prog && prog.result) || {};
  const aq = S.userName ? `?annotator=${encodeURIComponent(S.userName)}` : '';
  const [rules, relations, jdata] = await Promise.all([
    api(`/api/rules/${S.currentFile.id}${aq}`),
    api(`/api/relations/${S.currentFile.id}${aq}`),
    api(`/api/judge-data/${S.currentFile.id}${aq}`),
  ]);
  if (Array.isArray(rules)) { S.rules = rules; sortRules(); }
  if (Array.isArray(relations)) S.relations = relations;
  normalizeJudgeData(jdata);
  S.focusedRuleId = null; S.focusedRelId = S.relSource = S.relTarget = null;
  renderViewer();
  renderRightPanel();
  refreshFileBadge(S.currentFile.id);

  const cleanDone = prog && prog.done && !prog.error;
  if (cleanDone) {
    S.toolMode = false;
    document.getElementById('judgeBtn').classList.remove('active');
    renderJudgeModal();
    if (res.errors && res.errors.length) console.warn('judge errors:', res.errors);
  } else {
    renderJudgeModal();   // keep the modal open and explain what happened
    const msg = (prog && prog.done && prog.error) ? prog.error
              : (prog && prog.missing) ? 'Lost contact with the run — reloaded the latest results; verify before re-running.'
              : 'Run is taking longer than expected — reloaded the latest results; verify before re-running.';
    setToolStatus('judgeToolStatus', msg, 'error');
  }
}

// ─── Keyboard ────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (S.judgeRunning) return;   // locked while a judge run is in flight
  const tag = document.activeElement?.tagName?.toLowerCase();
  const inText = tag === 'textarea' || tag === 'input';
  // ⌘? toggles the file-comment panel (works from inside the textarea too,
  // so the same chord opens and closes it). On most layouts ? arrives as
  // Shift+/ — accept both key spellings.
  if ((e.metaKey || e.ctrlKey) && (e.key === '?' || (e.key === '/' && e.shiftKey))) {
    e.preventDefault(); toggleComment(); return;
  }
  if (e.key === 'Escape' && document.getElementById('exportMenu')) { closeExportMenu(); return; }
  if (e.key === 'Escape' && document.getElementById('voteInspector')) { document.getElementById('voteInspector').remove(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    if (inText) return;
    if (S.mode === 'relation') {
      if (S.relSource && S.relTarget) { e.preventDefault(); addRelation(); }   // commit the edge
    } else if (S.selection) {
      e.preventDefault(); addHandRule();
    }
    return;
  }
  if (inText) {
    // Esc inside the judge/comment modal closes it; elsewhere it just blurs the field.
    if (e.key === 'Escape') {
      if (S.toolMode) toggleTool();
      else if (S.commentMode) toggleComment();
      else document.activeElement.blur();
    }
    return;
  }
  if (S.toolMode) {
    if (e.key === 'Escape') toggleTool();
    return;
  }
  if (S.commentMode) {
    if (e.key === 'Escape') toggleComment();
    return;
  }
  if (S.mode === 'relation') {
    // Don't let keys inside the relation-type <select> dropdown (Delete/Esc/etc.)
    // cancel or unfocus the edge — those are dropdown interactions, not shortcuts.
    if (tag === 'select') return;
    // Delete / Backspace cancels the in-progress relation (same as Esc).
    if ((e.key === 'Delete' || e.key === 'Backspace') && (S.relSource || S.relTarget)) {
      e.preventDefault(); clearRelBuild(); return;
    }
    if (e.key === 'Escape') {
      if (S.relSource || S.relTarget) clearRelBuild();
      else if (S.focusedRelId) { S.focusedRelId = null; renderViewer(); renderRightPanel(); }
    }
    return;   // j/k/d rule-nav don't apply while labeling relations
  }
  // Editing a rule's text (press E): Enter saves the re-selection (or aborts if none),
  // Esc cancels. Other keys are swallowed so the drag-to-select isn't disturbed.
  if (S.editingRuleId) {
    if (e.key === 'Enter')  { e.preventDefault(); commitRuleEdit(); }
    if (e.key === 'Escape') { e.preventDefault(); cancelRuleEdit(); }
    return;
  }
  // E starts "edit text" mode (Extract only) — the focused span greys out and the
  // next drag-select redefines/extends its text.
  if (S.mode === 'extract' && S.focusedRuleId && !e.metaKey && !e.ctrlKey && !e.altKey && e.key === 'e') {
    e.preventDefault(); startRuleEdit(); return;
  }
  // a/s/d/f SET the deontic tag (PROHIBITION/PRESCRIPTION/PERMISSION/PREFERENCE)
  // whenever a RULE is focused in LABEL mode. (Set, not toggle.) Delete = ⌫.
  const TAG_KEYS = { a: 0, s: 1, d: 2, f: 3 };
  if (S.mode === 'label' && S.focusedRuleId && !e.metaKey && !e.ctrlKey && !e.altKey && (e.key in TAG_KEYS)) {
    const fr = S.rules.find(x => x.id === S.focusedRuleId);
    if (fr && ruleKind(fr) === 'rule') {
      e.preventDefault();
      setTag(TAGS[TAG_KEYS[e.key]], false);   // false = set/switch (don't toggle off)
      return;
    }
  }
  switch (e.key) {
    case 'j': case 'ArrowDown': e.preventDefault(); moveFocus(+1); break;
    case 'k': case 'ArrowUp':   e.preventDefault(); moveFocus(-1); break;
    case 'Delete': case 'Backspace':           // delete the focused rule/context
      if (S.focusedRuleId) { e.preventDefault(); deleteFocusedRule(); }
      break;
    case 'Escape':
      if (S.focusedRuleId) exitInspector();   // collapse the expanded rule
      else clearSelection();
      break;
  }
});

// ─── Export ──────────────────────────────────────────────────
// ⬇ CSV opens a small scope chooser: export just the current file, or all
// labeled files (both scoped to the active annotator).
function exportCSV(ev) {
  ev?.stopPropagation();
  if (document.getElementById('exportMenu')) { closeExportMenu(); return; }  // toggle
  const btn = document.querySelector('.csv-btn');
  const r = btn.getBoundingClientRect();
  const fileName = S.currentFile
    ? (S.currentFile.repo_name?.split('/').pop()
       || (S.currentFile.source_url || S.currentFile.id).split('/').pop())
    : null;
  const fileRules = S.currentFile ? S.rules.length : 0;
  const menu = document.createElement('div');
  menu.id = 'exportMenu';
  menu.className = 'export-menu';
  menu.style.top = (r.bottom + 6) + 'px';
  menu.style.right = Math.max(8, window.innerWidth - r.right) + 'px';
  menu.innerHTML = `
    <div class="export-menu-head">Export CSV${S.userName ? ' — ' + esc(S.userName) : ''}</div>
    <button class="export-opt" ${S.currentFile ? '' : 'disabled'} onclick="doExport('file')">
      📄 Current file${S.currentFile ? ` <small title="${esc(fileName)}">${esc(fileName)} · ${fileRules}</small>` : ' <small>none open</small>'}
    </button>
    <button class="export-opt" onclick="doExport('all')">🗂️ All labeled files</button>`;
  document.body.appendChild(menu);
  document.addEventListener('mousedown', closeExportMenuOnOutside);
}

function closeExportMenu() {
  document.getElementById('exportMenu')?.remove();
  document.removeEventListener('mousedown', closeExportMenuOnOutside);
}

function closeExportMenuOnOutside(e) {
  const m = document.getElementById('exportMenu');
  if (!m) { document.removeEventListener('mousedown', closeExportMenuOnOutside); return; }
  if (m.contains(e.target) || e.target.closest?.('.csv-btn')) return;
  closeExportMenu();
}

function doExport(scope) {
  const params = new URLSearchParams();
  if (S.userName) params.set('annotator', S.userName);
  if (scope === 'file' && S.currentFile) params.set('file_id', S.currentFile.id);
  closeExportMenu();
  window.open('/export' + (params.toString() ? '?' + params.toString() : ''), '_blank');
}

// ─── Helpers ─────────────────────────────────────────────────
async function api(url, method = 'GET', body = null) {
  try {
    const opts = { method, headers: {} };
    if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    const r = await fetch(url, opts); return r.json();
  } catch (e) { return { error: e.message }; }
}
function refreshFileBadge(id) {
  if (!id) return;
  const f = S.allFiles.find(f => f.id === id);
  if (f) {
    f.hand_count = S.rules.filter(r => r.source === 'hand').length;
    f.llm_count  = S.rules.filter(r => r.source !== 'hand').length;  // llm + revise
    f.revised    = S.rules.some(r => r.source === 'revise');         // green badge once revised
  }
  filterFiles(document.getElementById('fileSearch').value);
}
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function escHtml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function escAttr(s) { return s.replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;'); }
function fmtSize(n) { return n > 1024 ? `${(n / 1024).toFixed(1)}k` : `${n || '?'}b`; }

init();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5002")))
    parser.add_argument("--debug", action="store_true",
                        default=os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"))
    args = parser.parse_args()
    print(f"Rule Annotator → http://{args.host}:{args.port}")
    # threaded=True so a slow request (e.g. /export) never blocks the UI;
    # reloader stays off unless --debug so the container keeps running.
    app.run(debug=args.debug, host=args.host, port=args.port,
            threaded=True, use_reloader=args.debug)
