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

import hashlib, json, os, re, urllib.request, urllib.error
from datetime import datetime, timezone
import duckdb
from flask import Flask, jsonify, request

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
# ---------------------------------------------------------------------------
def annot_con():
    con = duckdb.connect(ANNOT_DB)
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
    for col in ("notes VARCHAR", "extracted_by VARCHAR"):
        try: con.execute(f"ALTER TABLE extracted_rules ADD COLUMN {col}")
        except Exception: pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_runs (
            id           VARCHAR PRIMARY KEY,
            file_id      VARCHAR NOT NULL,
            prompt       TEXT,
            model        VARCHAR,
            raw_response TEXT,
            rule_count   INTEGER DEFAULT 0,
            created_at   TIMESTAMP DEFAULT now()
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS saved_prompts (
            id        VARCHAR PRIMARY KEY,
            prompt    TEXT NOT NULL,
            label     VARCHAR,
            saved_at  TIMESTAMP DEFAULT now()
        )
    """)
    return con

_sample_ro = None
def sample_con():
    global _sample_ro
    if _sample_ro is None and os.path.exists(SAMPLE_DB):
        _sample_ro = duckdb.connect(SAMPLE_DB, read_only=True)
    return _sample_ro

def make_id(*parts):
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Routes – files
# ---------------------------------------------------------------------------
@app.route("/")
def index(): return HTML

@app.route("/api/files")
def api_files():
    sc = sample_con()
    if not sc:
        return jsonify({"error": "sample.db not found – run extract_sample.py first"}), 404
    rows = sc.execute(
        "SELECT id, source_url, repo_name, file_type, content_len, source "
        "FROM sample ORDER BY content_len DESC"
    ).fetchall()
    con = annot_con()
    raw = con.execute(
        "SELECT file_id, source, COUNT(*) FROM extracted_rules GROUP BY 1,2"
    ).fetchall()
    con.close()
    counts = {}
    for fid, src, n in raw:
        counts.setdefault(fid, {})[src] = n
    return jsonify([{
        "id": r[0], "source_url": r[1], "repo_name": r[2],
        "file_type": r[3], "content_len": r[4], "source": r[5],
        "hand_count": counts.get(r[0], {}).get("hand", 0),
        "llm_count":  counts.get(r[0], {}).get("llm",  0),
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
    sc.close()
    return jsonify([{
        "id": r[0], "file_id": r[1],
        "source_url": url_map.get(r[1], ""),
        "rule_text": r[2], "line_start": r[3], "line_end": r[4],
        "source": r[5], "extracted_by": r[6], "notes": r[7], "created_at": str(r[8]),
    } for r in rows])

@app.route("/api/rules/<fid>")
def api_rules(fid):
    con = annot_con()
    rows = con.execute("""
        SELECT id, rule_text, char_start, char_end, line_start, line_end,
               source, llm_run_id, notes, extracted_by, created_at
        FROM extracted_rules WHERE file_id=?
        ORDER BY COALESCE(char_start,999999), created_at
    """, [fid]).fetchall()
    con.close()
    return jsonify([_rule_row(r) for r in rows])

@app.route("/api/rules", methods=["POST"])
def save_rule():
    b = request.json or {}
    fid       = b.get("file_id","").strip()
    rule_text = b.get("rule_text","").strip()
    if not fid or not rule_text:
        return jsonify({"error": "file_id and rule_text required"}), 400
    rid = make_id(fid, "hand", rule_text)
    by  = b.get("extracted_by") or "unknown"
    con = annot_con()
    con.execute("""
        INSERT OR IGNORE INTO extracted_rules
            (id,file_id,rule_text,char_start,char_end,line_start,line_end,
             source,llm_run_id,notes,extracted_by)
        VALUES(?,?,?,?,?,?,?,'hand',NULL,NULL,?)
    """, [rid, fid, rule_text,
          b.get("char_start"), b.get("char_end"),
          b.get("line_start"), b.get("line_end"), by])
    con.close()
    return jsonify({
        "id": rid, "rule_text": rule_text,
        "char_start": b.get("char_start"), "char_end": b.get("char_end"),
        "line_start": b.get("line_start"), "line_end": b.get("line_end"),
        "source": "hand", "llm_run_id": None, "notes": None,
        "extracted_by": by,
    })

@app.route("/api/rules/<rid>", methods=["PATCH"])
def patch_rule(rid):
    b = request.json or {}
    notes = b.get("notes") or None
    con = annot_con()
    con.execute("UPDATE extracted_rules SET notes=? WHERE id=?", [notes, rid])
    con.close()
    return jsonify({"ok": True})

@app.route("/api/rules/<rid>", methods=["DELETE"])
def delete_rule(rid):
    con = annot_con()
    con.execute("DELETE FROM extracted_rules WHERE id=?", [rid])
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Routes – LLM
# ---------------------------------------------------------------------------
@app.route("/api/llm", methods=["POST"])
def run_llm():
    b       = request.json or {}
    fid     = b.get("file_id","").strip()
    prompt  = b.get("prompt","").strip()
    model   = b.get("model","gpt-4o-mini").strip()
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
            "You are a precise rule extractor. Extract all behavioral rules, constraints, "
            "guidelines, and instructions from the given file. "
            "Return ONLY a valid JSON array, no other text. "
            'Each element: {"rule_text":"<exact or near-exact quote>","line_start":<int>,"line_end":<int>}'
        )
        usr_msg = (
            f"{prompt}\n\nFile content:\n---\n{content[:30000]}\n---\n\nReturn ONLY a JSON array."
        )

    # Third-party models (anthropic/*, openai/*) use the Agent API; native Sonar models use Chat API
    use_agent_api = "/" in model
    if use_agent_api:
        payload_dict = {"model": model, "input": usr_msg}
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
            "temperature": 0.1, "max_tokens": 4096,
        }).encode()
        url = PERPLEXITY_CHAT_URL

    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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
        # Classification mode: store raw JSON, don't create extracted_rules rows
        con.execute(
            "INSERT INTO llm_runs(id,file_id,prompt,model,raw_response,rule_count) VALUES(?,?,?,?,?,?)",
            [run_id, fid, prompt, model, raw, 0]
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
        if not rt: continue
        # 1. Exact match
        idx = content.find(rt)
        # 2. Case-insensitive match
        if idx == -1: idx = cl.find(rt.lower())
        if idx >= 0:
            rule["char_start"] = idx
            rule["char_end"]   = idx + len(rt)
            rule["line_start"] = content[:idx].count("\n") + 1
            rule["line_end"]   = content[:idx+len(rt)].count("\n") + 1
        elif not rule.get("line_start"):
            # 3. Word-overlap fuzzy fallback: find line with most shared words
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

    con.execute(
        "INSERT INTO llm_runs(id,file_id,prompt,model,raw_response,rule_count) VALUES(?,?,?,?,?,?)",
        [run_id, fid, prompt, model, raw, len(extracted)]
    )
    saved = []
    for rule in extracted:
        rt = rule.get("rule_text","").strip()
        if not rt: continue
        eid = make_id(fid, "llm", run_id, rt)
        con.execute("""
            INSERT OR IGNORE INTO extracted_rules
                (id,file_id,rule_text,char_start,char_end,line_start,line_end,
                 source,llm_run_id,notes,extracted_by)
            VALUES(?,?,?,?,?,?,?,'llm',?,NULL,?)
        """, [eid, fid, rt,
              rule.get("char_start"), rule.get("char_end"),
              rule.get("line_start"), rule.get("line_end"),
              run_id, model])
        saved.append({
            "id": eid, "rule_text": rt,
            "char_start": rule.get("char_start"), "char_end": rule.get("char_end"),
            "line_start": rule.get("line_start"), "line_end": rule.get("line_end"),
            "source": "llm", "llm_run_id": run_id, "notes": None,
            "extracted_by": model,
        })
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
# Routes – CSV export page
# ---------------------------------------------------------------------------
@app.route("/export")
def export_page():
    acon = annot_con()
    rows = acon.execute("""
        SELECT r.file_id, r.rule_text, r.line_start, r.line_end,
               r.source, r.extracted_by, r.notes, r.created_at
        FROM extracted_rules r
        ORDER BY r.file_id, COALESCE(r.line_start, 999999), r.created_at
    """).fetchall()
    acon.close()
    sc = sample_con()
    url_map = {}
    if rows:
        fids = list({r[0] for r in rows})
        placeholders = ",".join("?" * len(fids))
        url_map = {r[0]: r[1] for r in sc.execute(
            f"SELECT id, source_url FROM sample WHERE id IN ({placeholders})", fids
        ).fetchall()}
    sc.close()

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["source_url","source","extracted_by","line_start","line_end","rule_text","notes","created_at"])
    for r in rows:
        writer.writerow([url_map.get(r[0],""), r[4], r[5] or "", r[2] or "", r[3] or "",
                         r[1], r[6] or "", str(r[7] or "")])
    csv_text = buf.getvalue()
    row_count = len(rows)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Rules Export</title>
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
  <h2>Rules Export</h2>
  <small>{row_count} rule{"s" if row_count!=1 else ""}</small>
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
  a.download = 'rules_export.csv'; a.click();
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
# Helpers
# ---------------------------------------------------------------------------
def _parse_llm_rules(text):
    text = re.sub(r'^```(?:json)?\s*','',text.strip())
    text = re.sub(r'\s*```$','',text).strip()
    try:
        d = json.loads(text)
        if isinstance(d, list): return d
    except: pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, list): return d
        except: pass
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
:root { --rules-panel-width: 420px; }
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

/* ── Viewer ── */
.viewer-panel {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  background: #fff;
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
.line-num {
  display: block;
  min-height: 1.6em;
  line-height: 1.6;
  border-radius: 4px;
  cursor: pointer;
  padding: 0 2px;
}
.line-num:hover { background: #eef2ff; color: #6366f1; }
.line-num.active { background: #e0e7ff; color: #4338ca; font-weight: 700; }
.content-pre {
  flex: 1; padding: 14px 16px; margin: 0;
  white-space: pre-wrap; word-break: break-word;
  overflow: visible; background: transparent; cursor: text;
}

/* ── Line highlights ── */
.line-hl {
  display: inline;
  cursor: pointer;
  border-radius: 0 3px 3px 0;
  transition: background 0.1s;
}
.line-hl:hover { filter: brightness(0.92); }

.panel-resizer {
  width: 8px; flex-shrink: 0; cursor: col-resize;
  background: linear-gradient(to right, #e6e6f3 0, #f2f2fa 100%);
  border-left: 1px solid #e0e0ea;
  border-right: 1px solid #e0e0ea;
}
.panel-resizer:hover,
.panel-resizer.active { background: linear-gradient(to right, #c7d2fe 0, #e0e7ff 100%); }

/* ── Rules panel ── */
.rules-panel {
  width: var(--rules-panel-width); min-width: 280px; max-width: 760px;
  flex-shrink: 0; display: flex; flex-direction: column;
  overflow: hidden; background: #fafafd;
}

/* annotator name bar */
.name-bar {
  padding: 7px 12px; background: #f0f0fa; border-bottom: 1px solid #e0e0f0;
  display: flex; align-items: center; gap: 8px; flex-shrink: 0;
}
.name-bar label { font-size: 11px; color: #7a7aaa; white-space: nowrap; }
.name-input {
  flex: 1; padding: 3px 7px; border-radius: 5px; border: 1px solid #d0d0e8;
  background: white; font-size: 12px; color: #333;
}
.name-input:focus { outline: none; border-color: #6366f1; }

.rules-section { display: flex; flex-direction: column; overflow: hidden; }
.rules-section.hand-section { flex: 0 0 auto; max-height: 44%; border-bottom: 2px solid #e8e8f4; }
.rules-section.llm-section  { flex: 1; overflow: hidden; }
.section-head {
  padding: 7px 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.2px;
  display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
  background: #f4f4fc; border-bottom: 1px solid #e4e4f0;
}
.section-count {
  font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 10px;
  background: #e8e8f8; color: #6060a0;
}
.rules-list { overflow-y: auto; flex: 1; padding: 5px; }

/* rule card – border/bg via CSS variables */
.rule-card {
  padding: 7px 9px; border-radius: 6px;
  border-left: 3px solid var(--rc-bdr, #d0d0e8);
  background: var(--rc-bg, #f8f8fc);
  margin-bottom: 4px; cursor: pointer; font-size: 12px; line-height: 1.5;
  display: flex; align-items: flex-start; gap: 6px;
  transition: background 0.1s; word-break: break-word;
}
.rule-card:hover  { background: var(--rc-hov, #f0f0f8); }
.rule-card.focused {
  background: var(--rc-act, #e8e8f4);
  box-shadow: 0 0 0 1.5px var(--rc-bdr, #9090c0);
}
.rule-card.no-pos { opacity: 0.75; }
.extractor-tag {
  display: inline-block; font-size: 9px; font-weight: 700; padding: 1px 5px;
  border-radius: 8px; background: var(--rc-bg2, #e8e8f4); color: var(--rc-bdr, #6060a0);
  margin-bottom: 3px; letter-spacing: 0.2px; vertical-align: middle;
}
.rule-body { flex: 1; min-width: 0; }
.rule-preview { color: #2a2a3e; margin-bottom: 2px; }
.rule-pos { font-size: 10px; color: #9090b0; }
.rule-note-text { margin-top: 4px; font-size: 11px; color: #6060a0; font-style: italic; white-space: pre-wrap; }
.rule-note-editor { display: none; margin-top: 5px; }
.rule-note-editor.open { display: block; }
.rule-note-input {
  width: 100%; padding: 4px 7px; border-radius: 5px;
  border: 1.5px solid #6366f1; background: white; font-size: 11px;
  font-family: inherit; color: #333; resize: none; line-height: 1.4;
}
.rule-note-input:focus { outline: none; }
.note-hint { font-size: 10px; color: #aaa; margin-top: 2px; }
.rule-actions { display: flex; flex-direction: column; gap: 2px; flex-shrink: 0; }
.rule-btn {
  background: none; border: none; cursor: pointer; padding: 2px 3px;
  border-radius: 3px; font-size: 13px; line-height: 1; color: #c0c0d8;
}
.rule-btn:hover { background: rgba(0,0,0,.07); color: #555; }
.rule-btn.del:hover { color: #e53e3e; background: rgba(229,62,62,.1); }
.rule-btn.note-btn.has-note { color: #6366f1; }

/* empty states */
.empty-hint { color: #b0b0c8; font-size: 12px; text-align: center; padding: 14px 8px; }
.viewer-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #b0b0c8; font-size: 14px; }

/* LLM controls */
.llm-controls {
  padding: 8px 11px; border-bottom: 1px solid #e8e8f4;
  flex-shrink: 0; display: flex; flex-direction: column; gap: 6px;
}
.llm-row { display: flex; gap: 6px; align-items: center; }
.llm-model {
  flex: 1; padding: 5px 7px; border-radius: 6px; border: 1px solid #d0d0e0;
  background: white; font-size: 12px; color: #333;
}
.llm-prompt {
  width: 100%; min-height: 60px; padding: 6px 8px; border-radius: 6px;
  border: 1px solid #d0d0e0; background: white; font-size: 12px; color: #333;
  resize: vertical; font-family: inherit; line-height: 1.4;
}
.extract-prompt-wrap { display: block; }
.extract-prompt-wrap.collapsed { display: none; }
.llm-prompt:focus, .llm-model:focus, .llm-label:focus { outline: none; border-color: #6366f1; }
.run-btn {
  padding: 5px 14px; border-radius: 20px; border: none; background: #6366f1;
  color: white; font-size: 12px; font-weight: 600; cursor: pointer; flex-shrink: 0;
}
.run-btn:hover:not(:disabled) { background: #4f46e5; }
.run-btn:disabled { opacity: 0.5; cursor: default; }
.icon-btn {
  padding: 4px 8px; border-radius: 6px; border: 1px solid #d0d0e0;
  background: transparent; color: #7070a0; font-size: 11px; cursor: pointer;
  white-space: nowrap;
}
.icon-btn:hover { border-color: #6366f1; color: #4338ca; }
.icon-btn:disabled { opacity: 0.4; cursor: default; }
.llm-status { font-size: 11px; color: #7070a0; min-height: 15px; flex: 1; }
.llm-status.error { color: #e53e3e; }
.llm-status.ok    { color: #22c55e; }

/* prompt history */
.prompt-history {
  border: 1px solid #e0e0f0; border-radius: 6px; background: white;
  max-height: 120px; overflow-y: auto; display: none;
}
.prompt-history.open { display: block; }
.ph-item {
  padding: 5px 9px; font-size: 11px; color: #444; cursor: pointer;
  border-bottom: 1px solid #f0f0f8; display: flex; gap: 6px; align-items: flex-start;
}
.ph-item:last-child { border-bottom: none; }
.ph-item:hover { background: #f4f4fc; }
.ph-text { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ph-label { font-size: 10px; font-weight: 600; color: #6366f1; flex-shrink: 0; }
.ph-date  { font-size: 10px; color: #aaa; flex-shrink: 0; }
.ph-del   { font-size: 13px; color: #ccc; background: none; border: none; cursor: pointer; flex-shrink: 0; padding: 0; }
.ph-del:hover { color: #e53e3e; }

/* collapsible sub-sections */
.sub-head {
  padding: 6px 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.2px;
  display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none;
  background: #f0f0f8; border-bottom: 1px solid #e0e0ee; flex-shrink: 0;
}
.sub-head:hover { background: #e8e8f4; }
.sub-arrow { font-size: 9px; color: #8080a0; transition: transform 0.15s; display: inline-block; }
.sub-arrow.closed { transform: rotate(-90deg); }
.sub-body { display: flex; flex-direction: column; overflow: hidden; }
.sub-body.collapsed { display: none; }

/* classification result */
.classif-result {
  border-top: 1px solid #e4e4f0; overflow-y: auto; max-height: 55vh;
  font-size: 12px; flex-shrink: 0;
}
.cr-row {
  display: flex; gap: 8px; padding: 5px 12px; border-bottom: 1px solid #f0f0f8;
  align-items: flex-start; line-height: 1.4;
}
.cr-row:last-child { border-bottom: none; }
.cr-label {
  font-weight: 700; color: #6060a0; min-width: 80px; flex-shrink: 0; font-size: 11px;
  padding-top: 1px;
}
.cr-val { color: #222; flex: 1; white-space: pre-wrap; word-break: break-word; }
.cr-badge {
  display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 10px;
  font-weight: 700; white-space: nowrap;
}
.cr-badge.green  { background: #dcfce7; color: #15803d; }
.cr-badge.yellow { background: #fef9c3; color: #854d0e; }
.cr-badge.orange { background: #ffedd5; color: #c2410c; }
.cr-badge.red    { background: #fee2e2; color: #b91c1c; }
.cr-badge.blue   { background: #dbeafe; color: #1d4ed8; }
.cr-badge.gray   { background: #f1f5f9; color: #475569; }
.cr-toggle { cursor: pointer; color: #6366f1; font-size: 11px; padding: 4px 12px;
             background: none; border: none; display: block; width: 100%; text-align: left; }
.cr-json { display: none; padding: 8px 12px; background: #f8f8fc; }
.cr-json.open { display: block; }
.cr-json pre { margin: 0; font-size: 11px; white-space: pre-wrap; word-break: break-word; color: #333; }

/* bulk selection */
.rule-card.selected { outline: 2px solid #6366f1; outline-offset: -2px; }
.rule-checkbox { display: flex; align-items: flex-start; padding: 2px 4px 0 0; flex-shrink: 0; cursor: pointer; }
.rule-checkbox input { cursor: pointer; accent-color: #6366f1; }
.bulk-bar {
  display: none; align-items: center; gap: 6px; padding: 6px 11px;
  background: #eef2ff; border-bottom: 1px solid #c7d2fe; flex-shrink: 0; font-size: 12px;
}
.bulk-bar.visible { display: flex; }
.bulk-count { color: #4338ca; font-weight: 600; flex: 1; }
.rules-tools {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: #f7f8ff;
  border-bottom: 1px solid #e1e5ff;
  flex-shrink: 0;
}
.rules-search {
  flex: 1;
  min-width: 0;
  padding: 5px 8px;
  border-radius: 6px;
  border: 1px solid #cfd6ff;
  background: white;
  color: #2a2a3e;
  font-size: 12px;
}
.rules-search:focus {
  outline: none;
  border-color: #6366f1;
}
.rules-tools-count {
  font-size: 10px;
  color: #7070a0;
  min-width: 52px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.bulk-btn {
  padding: 3px 10px; border-radius: 12px; border: 1px solid #a5b4fc; background: white;
  color: #4338ca; font-size: 11px; font-weight: 600; cursor: pointer; white-space: nowrap;
}
.bulk-btn:hover { background: #eef2ff; border-color: #6366f1; }
.bulk-btn.danger { border-color: #fca5a5; color: #dc2626; }
.bulk-btn.danger:hover { background: #fff1f1; border-color: #f87171; }
.sel-all-btn {
  font-size: 10px; font-weight: 500; color: #8080b0; background: none; border: none;
  cursor: pointer; padding: 0 2px; text-decoration: underline dotted;
}
.sel-all-btn:hover { color: #4338ca; }

/* keyboard bar */
.kb-bar {
  padding: 5px 10px; background: #f0f0f8; border-top: 1px solid #e0e0ee;
  font-size: 10px; color: #9090b0; display: flex; gap: 8px; flex-wrap: wrap; flex-shrink: 0;
}
.kb-bar span { white-space: nowrap; }
.export-btn {
  margin-left: auto; padding: 2px 8px; border-radius: 10px;
  border: 1px solid #c0c0d8; background: white; color: #6060a0;
  font-size: 10px; font-weight: 600; cursor: pointer; white-space: nowrap;
}
.export-btn:hover { border-color: #6366f1; color: #4338ca; }
</style>
</head>
<body>
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
      <span class="sel-info" id="selInfo">
        <span class="kb">Select text then <kbd>⌘↵</kbd> to add &nbsp;·&nbsp;
        <kbd>j</kbd><kbd>k</kbd> nav &nbsp;·&nbsp; <kbd>n</kbd> note &nbsp;·&nbsp; <kbd>d</kbd> del</span>
      </span>
      <button class="add-btn" id="addBtn" disabled onclick="addHandRule()">+ Add Rule</button>
    </div>
    <div class="viewer-body" id="viewerBody">
      <div class="viewer-empty">← pick a file</div>
    </div>
  </div>

  <div class="panel-resizer" id="panelResizer" title="Drag to resize rules panel"></div>

  <!-- ── Rules panel ── -->
  <div class="rules-panel">

    <!-- annotator name -->
    <div class="name-bar">
      <label for="annotatorName">Annotator:</label>
      <input id="annotatorName" class="name-input" placeholder="Your name"
             oninput="saveName(this.value)">
    </div>

    <!-- bulk action bar -->
    <div class="bulk-bar" id="bulkBar">
      <span class="bulk-count" id="bulkCount">0 selected</span>
      <button class="bulk-btn danger" onclick="bulkDelete()">Delete</button>
      <button class="bulk-btn" onclick="bulkMerge()">Merge</button>
      <button class="bulk-btn" onclick="clearRuleSelection()">Clear</button>
    </div>
    <div class="rules-tools">
      <input id="ruleSearch" class="rules-search" placeholder="Search hand + LLM rules or line:42"
             oninput="onRuleSearchInput(this.value)">
      <button class="bulk-btn" id="clearRuleSearchBtn" onclick="clearRuleSearch()" disabled>Clear</button>
      <span class="rules-tools-count" id="rulesVisibleCount">0/0</span>
    </div>

    <!-- hand rules -->
    <div class="rules-section hand-section">
      <div class="section-head">
        <span>Hand Rules</span>
        <span class="section-count" id="handCount">0</span>
        <button class="sel-all-btn" onclick="selectAll('hand')">select all</button>
      </div>
      <div class="rules-list" id="handRules">
        <div class="empty-hint">Select text to add rules</div>
      </div>
    </div>

    <!-- LLM section (two collapsible sub-sections) -->
    <div class="rules-section llm-section">

      <!-- ── Extract Rules ── -->
      <div class="sub-head" onclick="toggleSub('extractBody','extractArrow')">
        <span class="sub-arrow" id="extractArrow">▾</span>
        <span>Extract Rules</span>
        <span class="section-count" id="llmCount">0</span>
        <button class="sel-all-btn" onclick="event.stopPropagation();selectAll('llm')">select all</button>
      </div>
      <div class="sub-body" id="extractBody">
        <div class="llm-controls">
          <div class="llm-row">
            <select class="llm-model" id="extractModel">
              <optgroup label="Perplexity">
                <option value="sonar" selected>sonar</option>
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
              </optgroup>
              <optgroup label="OpenAI (via Perplexity)">
                <option value="openai/gpt-5-mini">gpt-5-mini</option>
                <option value="openai/gpt-5">gpt-5</option>
                <option value="openai/gpt-5.1">gpt-5.1</option>
                <option value="openai/gpt-5.4-mini">gpt-5.4-mini</option>
                <option value="openai/gpt-5.4">gpt-5.4</option>
                <option value="openai/gpt-5.5">gpt-5.5</option>
              </optgroup>
            </select>
            <button class="run-btn" id="extractBtn" onclick="runExtraction()" disabled>Extract</button>
          </div>
          <div class="llm-row" style="justify-content:flex-end">
            <button class="icon-btn" id="extractPromptToggle" onclick="toggleExtractPrompt(event)">Hide prompt</button>
          </div>
          <div class="extract-prompt-wrap" id="extractPromptWrap">
          <textarea class="llm-prompt" id="extractPrompt"
            placeholder="Describe what to extract…">Extract all rule-like spans from this file for enforceability analysis.

A rule is any instruction, constraint, prohibition, or requirement given to an LLM or agent. Prefer rules that are specific and potentially checkable — for example:
- version or toolchain requirements ("Python 3.13.2 or compatible")
- file or artifact requirements ("log changes in CHANGELOG.md")
- workflow ordering constraints ("always follow this chain, never skip layers")
- code style or format mandates ("follow PEP 8")
- procedural gates ("read all memory bank files at the start of every task")
- naming, header, or metadata requirements ("bump @version in the userscript header")

Prefer exact or near-exact quotes from the file. Extract each independently enforceable clause as its own rule — do not merge unrelated requirements. Skip purely motivational or explanatory text with no checkable obligation.

Return only a valid JSON array. Each element:
{"rule_text": "<exact or near-exact quote>", "line_start": <int>, "line_end": <int>}</textarea>
          </div>
          <div class="llm-row">
            <span class="llm-status" id="extractStatus"></span>
            <button class="icon-btn" id="clearExtractBtn" onclick="clearExtraction()" disabled style="margin-left:auto">Clear LLM</button>
          </div>
        </div>
        <div class="rules-list" id="llmRules">
          <div class="empty-hint">Run extraction to find rules</div>
        </div>
      </div>

      <!-- ── Classify Rule ── -->
      <div class="sub-head" onclick="toggleSub('classifyBody','classifyArrow')">
        <span class="sub-arrow" id="classifyArrow">▾</span>
        <span>Classify Rule</span>
      </div>
      <div class="sub-body" id="classifyBody">
      <div class="llm-controls">
        <div class="llm-row">
          <select class="llm-model" id="llmModel">
            <optgroup label="Perplexity">
              <option value="sonar">sonar</option>
              <option value="sonar-pro">sonar-pro</option>
              <option value="sonar-reasoning-pro">sonar-reasoning-pro</option>
              <option value="sonar-deep-research">sonar-deep-research</option>
            </optgroup>
            <optgroup label="Anthropic (via Perplexity)">
              <option value="anthropic/claude-haiku-4-5">claude-haiku-4-5</option>
              <option value="anthropic/claude-sonnet-4-5">claude-sonnet-4-5</option>
              <option value="anthropic/claude-sonnet-4-6" selected>claude-sonnet-4-6</option>
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
            </optgroup>
          </select>
          <button class="run-btn" id="runBtn" onclick="runLLM()" disabled>Classify Rule</button>
        </div>
        <textarea class="llm-prompt" id="llmPrompt"
          placeholder="Describe what to extract…"
          oninput="updateRunBtn()">You are classifying rules from LLM system prompts according to whether and how they can be deterministically enforced.

A "rule" is a natural-language instruction, constraint, prohibition, or requirement given to an LLM or agent. Your task is not to judge whether the rule is good, but to determine what would be needed to enforce it outside the model.

Classify each rule independently. Do not let surrounding rules make a vague rule enforceable unless the needed definition is explicitly present in the provided context.

Classify the following rule:

RULE: {rule text}

OPTIONAL CONTEXT:
{insert surrounding prompt, repo context, tool list, or project assumptions here}

Use the taxonomy below.

## 1. Prerequisite information

Classify what information is needed to enforce the rule.

Choose one or more:

* `none`: The rule can be checked without external context.
* `static_prompt_context`: The rule depends only on the prompt text or declared policy text.
* `session_state`: The rule depends on current conversation/session state.
* `repo_state`: The rule depends on files, directory structure, package configuration, git state, or project metadata.
* `runtime_state`: The rule depends on process state, command results, environment variables, logs, network state, or tool outputs.
* `external_policy_state`: The rule depends on settings outside the repo/session, such as GitHub branch protection, PR settings, deployment rules, organization policy, or access-control state.
* `learned_project_knowledge`: The rule depends on project conventions, architecture, naming schemes, or "how this codebase works" knowledge that is not explicitly represented.
* `user_defined_terms`: The rule contains terms that need user definition before enforcement, such as "important," "safe," "appropriate," "large," "production," "sensitive," or "simple."
* `generated_information`: Enforcing the rule requires deriving or generating additional facts, summaries, plans, classifications, or semantic labels.
* `pre_action_procedure`: Enforcing the rule requires running a procedure before an action, such as tests, search, dependency analysis, approval lookup, or impact analysis.
* `self_referential_or_open_ended`: The rule refers to broad, evolving, or self-referential knowledge such as "follow all project conventions," "use best practices," or "respect prior decisions."

For each selected prerequisite, explain briefly what information is required.

## 2. Enforcement mechanism

Classify what kind of checker could enforce the rule.

Choose one primary mechanism and any secondary mechanisms:

* `literal_match`: Exact string, token, filename, command, or value matching.
* `regex`: Regular-expression matching.
* `schema_or_type_check`: JSON schema, API schema, type system, config schema, or static structural validation.
* `linter_or_static_analysis`: Code linter, AST check, dependency check, import check, or static program analysis.
* `bash_or_script`: Shell command, script, grep, git command, filesystem inspection, test command, or CI check.
* `unit_or_integration_test`: Deterministic test execution.
* `policy_engine`: Declarative rule engine, access-control engine, Datalog, OPA/Rego, SQL query, or provenance/data-flow checker.
* `runtime_monitor`: Post-execution monitor over logs, tool calls, filesystem changes, network calls, or process behavior.
* `human_review_gate`: Requires human judgment or approval.
* `llm_judge`: Requires semantic interpretation by an LLM or other learned model.
* `other`: other system/software mechanism that could enforce this rule
* `not_enforceable_deterministically`: Cannot be reliably enforced with deterministic machinery as stated.

Explain why the chosen mechanism is sufficient or insufficient.

## 3. Enforcement trigger / location

Classify when or where enforcement should happen.

Choose one or more:

* `session_init`: Before the agent starts; e.g., load rules, check environment, read repo state.
* `settings_or_config`: At configuration time; e.g., settings.json, policy file, repo config, CI config.
* `pre_action_gate`: Before a tool call, shell command, file edit, API call, commit, PR, deployment, or message send.
* `intermediate_output`: While the agent is producing plans, summaries, code patches, or partial results.
* `post_exec`: After a command/tool/action executes, checking effects or logs.
* `verify_gate`: Before declaring success, merging, submitting, deploying, or finalizing.
* `final_output`: Before the final answer is shown to the user.

Explain the earliest reliable trigger and the latest safe trigger.

## 4. Ambiguity

Assess the ambiguity of the rule.

Choose one:

* `none`: Fully specified and directly checkable.
* `low`: Minor assumptions needed; likely enforceable after normalizing terms.
* `medium`: Important terms or thresholds need definition.
* `high`: Rule relies heavily on intent, judgment, project conventions, or unstated context.
* `not_actionable`: Too vague to enforce as written.

Then list:

* ambiguous terms
* missing thresholds or definitions
* assumptions an enforcer would need
* questions to ask the user to make the rule enforceable

## 5. Deterministic enforceability

Classify the rule's deterministic enforceability.

Choose one:

* `deterministically_enforceable_as_written`
* `deterministically_enforceable_with_context`
* `deterministically_enforceable_after_rewrite`
* `partially_enforceable`
* `requires_llm_or_human_judgment`
* `not_enforceable`

Explain the classification.

## 6. Suggested enforceable rewrite

Rewrite the rule into one or more enforceable rules.

Prefer rules that are:

* observable
* checkable
* scoped to a specific trigger
* explicit about required inputs
* explicit about pass/fail criteria

If the original rule is vague, split it into:

1. a deterministic core rule
2. a semantic or human-review residual rule

## Output format

Return valid JSON with this structure:

{
"rule": "...",
"summary": "...",
"prerequisites": [
{
"type": "...",
"details": "..."
}
],
"enforcement_mechanism": {
"primary": "...",
"secondary": ["..."],
"details": "..."
},
"trigger_location": {
"recommended": ["..."],
"earliest_reliable": "...",
"latest_safe": "...",
"details": "..."
},
"ambiguity": {
"level": "...",
"ambiguous_terms": ["..."],
"missing_definitions": ["..."],
"assumptions_needed": ["..."],
"questions_for_user": ["..."]
},
"deterministic_enforceability": {
"classification": "...",
"details": "..."
},
"suggested_enforceable_rewrite": {
"deterministic_core": ["..."],
"semantic_or_review_residual": ["..."]
},
"confidence": "low | medium | high"
}</textarea>
        <div class="llm-row" style="gap:5px">
          <button class="icon-btn" onclick="savePrompt()">💾 Save</button>
          <button class="icon-btn" id="historyBtn" onclick="toggleHistory()">🕑 History</button>
          <button class="icon-btn" id="clearRunBtn" onclick="clearLastRun()" disabled style="margin-left:auto">Clear run</button>
        </div>
        <div class="prompt-history" id="promptHistory"></div>
        <div class="llm-row">
          <span class="llm-status" id="llmStatus"></span>
        </div>
      </div>
      <div class="classif-result" id="classifResult" style="display:none"></div>
      <div class="rules-list" id="classifyRulesHint">
        <div class="empty-hint">Run the LLM to extract rules</div>
      </div>
    </div>

    <div class="kb-bar">
      <span><kbd>⌘↵</kbd> add</span>
      <span><kbd>j</kbd><kbd>k</kbd> nav</span>
      <span><kbd>n</kbd> note</span>
      <span><kbd>d</kbd> del</span>
      <span><kbd>x</kbd> select</span>
      <span><kbd>↵</kbd> save note</span>
      <span><kbd>Esc</kbd> cancel</span>
      <button class="export-btn" onclick="exportCSV()" title="Export rules as CSV">⬇ CSV</button>
    </div>
  </div>
</div>

<script>
// ─── State ───────────────────────────────────────────────────
const S = {
  allFiles:      [],
  currentFile:   null,
  rules:         [],
  selection:     null,
  focusedRuleId: null,
  editingNoteId: null,
  lastRunId:        null,
  lastExtractRunId: null,
  userName:      localStorage.getItem('annotatorName') || '',
  selectedIds:   new Set(),
  lastSelectedId: null,
  ruleSearchQuery: '',
  extractPromptCollapsed: localStorage.getItem('extractPromptCollapsed') === '1',
};
const RULES_PANEL_WIDTH_KEY = 'annotator.rulesPanelWidth';
const RULES_PANEL_MIN = 280;
const RULES_PANEL_MAX = 760;

// ─── Colour palette (per-extractor) ──────────────────────────
// Each entry: bg, bdr (border), hov (hover bg), act (active bg), bg2 (tag bg)
const COLORS = [
  { bg:'rgba(245,158,11,.18)', bdr:'#f59e0b', hov:'rgba(245,158,11,.34)', act:'rgba(245,158,11,.52)', bg2:'rgba(245,158,11,.22)' },
  { bg:'rgba(99,102,241,.17)', bdr:'#6366f1', hov:'rgba(99,102,241,.32)', act:'rgba(99,102,241,.48)', bg2:'rgba(99,102,241,.21)' },
  { bg:'rgba(16,185,129,.17)', bdr:'#10b981', hov:'rgba(16,185,129,.32)', act:'rgba(16,185,129,.48)', bg2:'rgba(16,185,129,.21)' },
  { bg:'rgba(239,68,68,.17)',  bdr:'#ef4444', hov:'rgba(239,68,68,.32)',  act:'rgba(239,68,68,.48)',  bg2:'rgba(239,68,68,.21)'  },
  { bg:'rgba(236,72,153,.17)', bdr:'#ec4899', hov:'rgba(236,72,153,.32)', act:'rgba(236,72,153,.48)', bg2:'rgba(236,72,153,.21)' },
  { bg:'rgba(14,165,233,.17)', bdr:'#0ea5e9', hov:'rgba(14,165,233,.32)', act:'rgba(14,165,233,.48)', bg2:'rgba(14,165,233,.21)' },
  { bg:'rgba(168,85,247,.17)', bdr:'#a855f7', hov:'rgba(168,85,247,.32)', act:'rgba(168,85,247,.48)', bg2:'rgba(168,85,247,.21)' },
  { bg:'rgba(251,146,60,.17)', bdr:'#fb923c', hov:'rgba(251,146,60,.32)', act:'rgba(251,146,60,.48)', bg2:'rgba(251,146,60,.21)' },
];
const _cc = {};
function colorFor(name) {
  if (!name) name = 'unknown';
  if (_cc[name]) return _cc[name];
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return (_cc[name] = COLORS[h % COLORS.length]);
}
function cssVars(col) {
  return `--hl-bg:${col.bg};--hl-bdr:${col.bdr};--hl-hov:${col.hov};--hl-act:${col.act}`;
}
function rcVars(col) {
  return `--rc-bdr:${col.bdr};--rc-bg:${col.bg};--rc-hov:${col.hov};--rc-act:${col.act};--rc-bg2:${col.bg2}`;
}

function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

function maxRulesWidth() {
  const layout = document.querySelector('.layout');
  const filePanel = document.querySelector('.file-panel');
  if (!layout || !filePanel) return RULES_PANEL_MAX;
  const minViewerWidth = 340;
  const resizerWidth = 8;
  const available = Math.floor(layout.clientWidth - filePanel.clientWidth - minViewerWidth - resizerWidth);
  return Math.max(RULES_PANEL_MIN, Math.min(RULES_PANEL_MAX, available));
}

function setRulesPanelWidth(px, save=true) {
  const width = clamp(Math.round(px), RULES_PANEL_MIN, maxRulesWidth());
  document.documentElement.style.setProperty('--rules-panel-width', `${width}px`);
  if (save) localStorage.setItem(RULES_PANEL_WIDTH_KEY, String(width));
}

function initResizableRulesPanel() {
  const saved = parseInt(localStorage.getItem(RULES_PANEL_WIDTH_KEY) || '', 10);
  if (!Number.isNaN(saved)) setRulesPanelWidth(saved, false);
  else setRulesPanelWidth(420, false);

  const resizer = document.getElementById('panelResizer');
  const layout = document.querySelector('.layout');
  if (!resizer || !layout) return;

  function onMove(e) {
    const rect = layout.getBoundingClientRect();
    setRulesPanelWidth(rect.right - e.clientX);
  }
  function stopResize() {
    document.body.classList.remove('resizing');
    resizer.classList.remove('active');
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', stopResize);
  }
  resizer.addEventListener('mousedown', (e) => {
    e.preventDefault();
    document.body.classList.add('resizing');
    resizer.classList.add('active');
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', stopResize);
  });
  window.addEventListener('resize', () => {
    const current = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--rules-panel-width') || '', 10);
    if (!Number.isNaN(current)) setRulesPanelWidth(current, false);
  });
}

// ─── Boot ────────────────────────────────────────────────────
async function init() {
  document.getElementById('annotatorName').value = S.userName;
  initResizableRulesPanel();
  updateRunBtn();
  loadPromptHistory();
  updateExtractPromptUI();
  const files = await api('/api/files');
  if (files.error) { setStatus(files.error,'error'); return; }
  S.allFiles = files;
  document.getElementById('fileCountLabel').textContent = files.length;
  renderFileList(files);
}

// ─── Name ────────────────────────────────────────────────────
function saveName(v) {
  S.userName = v.trim();
  localStorage.setItem('annotatorName', S.userName);
}

function updateExtractPromptUI() {
  const wrap = document.getElementById('extractPromptWrap');
  const btn = document.getElementById('extractPromptToggle');
  if (!wrap || !btn) return;
  wrap.classList.toggle('collapsed', S.extractPromptCollapsed);
  btn.textContent = S.extractPromptCollapsed ? 'Show prompt' : 'Hide prompt';
}

function toggleExtractPrompt(evt) {
  evt?.stopPropagation();
  S.extractPromptCollapsed = !S.extractPromptCollapsed;
  localStorage.setItem('extractPromptCollapsed', S.extractPromptCollapsed ? '1' : '0');
  updateExtractPromptUI();
}

// ─── File list ───────────────────────────────────────────────
function renderFileList(files) {
  const el = document.getElementById('fileList');
  if (!files.length) { el.innerHTML = '<div class="empty-hint">No files</div>'; return; }
  // Get global index by id
  const idxMap = {};
  S.allFiles.forEach((f,i) => idxMap[f.id] = i+1);
  el.innerHTML = files.map(f => {
    const name = f.repo_name
      ? f.repo_name.split('/').pop()
      : (f.source_url||f.id).split('/').pop() || f.id;
    const hCol = f.hand_count ? colorFor(S.userName || 'hand') : null;
    const lCol = f.llm_count  ? colorFor(document.getElementById('llmModel').value) : null;
    const badges = [
      hCol ? `<span class="badge" style="background:${hCol.bg2};color:${hCol.bdr}">✎ ${f.hand_count}</span>` : '',
      lCol ? `<span class="badge" style="background:${lCol.bg2};color:${lCol.bdr}">⚡ ${f.llm_count}</span>`  : '',
    ].filter(Boolean).join('');
    const active = S.currentFile?.id === f.id ? ' active' : '';
    return `<div class="file-item${active}" id="fi-${f.id}" onclick="selectFile('${f.id}')">
      <div class="file-idx">${idxMap[f.id]||'?'}</div>
      <div class="file-info">
        <div class="file-name" title="${esc(f.source_url||f.id)}">${esc(name)}</div>
        <div class="file-meta">
          <span>${esc(f.file_type||'—')}</span>
          <span>${fmtSize(f.content_len)}</span>
          ${badges}
        </div>
      </div>
    </div>`;
  }).join('');
}

function filterFiles(q) {
  q = q.toLowerCase();
  renderFileList(q
    ? S.allFiles.filter(f =>
        (f.repo_name||'').toLowerCase().includes(q) ||
        (f.source_url||'').toLowerCase().includes(q) ||
        (f.file_type||'').toLowerCase().includes(q))
    : S.allFiles);
}

function getRuleLineRange(rule) {
  const lsRaw = rule.line_start;
  const leRaw = rule.line_end != null ? rule.line_end : rule.line_start;
  const hasLineRange = lsRaw != null && lsRaw !== '' && leRaw != null && leRaw !== '';
  const ls = Number(lsRaw);
  const le = Number(leRaw);
  if (hasLineRange && Number.isFinite(ls) && Number.isFinite(le)) {
    const start = Math.max(1, Math.min(ls, le));
    const end = Math.max(start, Math.max(ls, le));
    return [start, end];
  }
  if (rule.char_start == null || !S.currentFile?.content) return null;
  const cs = Number(rule.char_start);
  const ce = Number(rule.char_end != null ? rule.char_end : rule.char_start);
  if (!Number.isFinite(cs) || !Number.isFinite(ce)) return null;
  const content = S.currentFile.content;
  const a = Math.max(0, Math.min(content.length, Math.min(cs, ce)));
  const b = Math.max(0, Math.min(content.length, Math.max(cs, ce)));
  const start = content.slice(0, a).split('\n').length;
  const end = content.slice(0, b).split('\n').length;
  return [start, Math.max(start, end)];
}

function lineRuleMatches(rule, lineNo) {
  const range = getRuleLineRange(rule);
  if (!range) return false;
  return lineNo >= range[0] && lineNo <= range[1];
}

function parseLineQuery(q) {
  const s = (q || '').trim().toLowerCase();
  if (!s) return null;
  const m = s.match(/^(?:line|l)\s*[:#]?\s*(\d+)$/) || s.match(/^#(\d+)$/) || s.match(/^(\d+)$/);
  if (!m) return null;
  const lineNo = parseInt(m[1], 10);
  return Number.isFinite(lineNo) && lineNo > 0 ? lineNo : null;
}

function ruleMatchesSearch(rule, q) {
  const needle = (q || '').trim().toLowerCase();
  if (!needle) return true;
  const lineNo = parseLineQuery(needle);
  if (lineNo != null) return lineRuleMatches(rule, lineNo);
  const hay = [
    rule.rule_text || '',
    rule.notes || '',
    rule.extracted_by || '',
    rule.source || '',
  ].join('\n').toLowerCase();
  return hay.includes(needle);
}

function getVisibleRules() {
  const q = S.ruleSearchQuery;
  if (!q || !q.trim()) return S.rules;
  return S.rules.filter(r => ruleMatchesSearch(r, q));
}

function setRuleSearchQuery(value, focusInput=false) {
  const q = value || '';
  S.ruleSearchQuery = q;
  const input = document.getElementById('ruleSearch');
  if (input && input.value !== q) input.value = q;
  if (focusInput && input) input.focus();
}

function applyLineSearch(lineNo) {
  const n = Number(lineNo);
  if (!Number.isFinite(n) || n < 1) return;
  setRuleSearchQuery(`line:${n}`);
  renderViewer();
  renderRulesPanel();
}

function onRuleSearchInput(v) {
  setRuleSearchQuery(v || '');
  renderRulesPanel();
  renderViewer();
}

function clearRuleSearch() {
  setRuleSearchQuery('', true);
  renderRulesPanel();
  renderViewer();
}

function lineNumberClick(lineNo, evt) {
  evt?.stopPropagation();
  clearSelection();
  applyLineSearch(lineNo);
}

// ─── Select file ─────────────────────────────────────────────
async function selectFile(id) {
  const [file, rules] = await Promise.all([
    api(`/api/file/${id}`), api(`/api/rules/${id}`),
  ]);
  if (file.error) { setStatus(file.error,'error'); return; }
  S.currentFile = file; S.rules = rules;
  S.selection = null; S.focusedRuleId = null; S.editingNoteId = null; S.selectedIds.clear();
  S.lastExtractRunId = null;
  setRuleSearchQuery('');
  document.getElementById('runBtn').disabled = false;
  document.getElementById('extractBtn').disabled = false;
  document.getElementById('addBtn').disabled = true;
  setSelInfo(null);
  renderFileList(S.allFiles);
  renderViewerHead(file);
  renderViewer();
  renderRulesPanel();
}

// ─── Viewer ──────────────────────────────────────────────────
function renderViewerHead(file) {
  const title = file.repo_name
    ? `${file.repo_name}  /  ${(file.source_url||'').split('/').pop()}`
    : (file.source_url||file.id).split('/').pop();
  document.getElementById('viewerHead').innerHTML = `
    <span class="viewer-title" title="${esc(file.source_url||'')}">${esc(title)}</span>
    ${file.source_url ? `<a href="${esc(file.source_url)}" target="_blank">↗ source</a>` : ''}
    <span style="font-size:11px;color:#aaa;flex-shrink:0">${fmtSize(file.content_len)}</span>
  `;
}

function renderViewer() {
  const body = document.getElementById('viewerBody');
  if (!S.currentFile) {
    body.innerHTML = '<div class="viewer-empty">← pick a file</div>'; return;
  }
  const content = S.currentFile.content;
  const lineCount = (content.match(/\n/g)||[]).length + 1;
  const searchLine = parseLineQuery(S.ruleSearchQuery);
  const lineNums = Array.from({length:lineCount}, (_,i) => {
    const lineNo = i + 1;
    const active = searchLine === lineNo ? ' active' : '';
    return `<span class="line-num${active}" onclick="lineNumberClick(${lineNo}, event)">${lineNo}</span>`;
  }).join('');
  body.innerHTML = `
    <div class="line-nums">${lineNums}</div>
    <pre class="content-pre" id="contentPre"></pre>
  `;
  document.getElementById('contentPre').innerHTML =
    renderLines(content, S.rules);
  const pre = document.getElementById('contentPre');
  pre.addEventListener('mouseup', onViewerMouseUp);
  pre.addEventListener('mouseover', e => {
    const el = e.target.closest?.('.line-hl[data-rids]');
    if (el && el.dataset.rids.split(',').every(id => id !== S.focusedRuleId))
      el.style.background = el.dataset.hov;
  });
  pre.addEventListener('mouseout', e => {
    const el = e.target.closest?.('.line-hl[data-rids]');
    if (el && el.dataset.rids.split(',').every(id => id !== S.focusedRuleId))
      el.style.background = el.dataset.bg;
  });
}

// Parse "rgba(r,g,b,a)" → [r,g,b,a]
function parseRgba(s) {
  const m = s.match(/rgba?\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),?\s*(\d*\.?\d+)?\)/);
  return m ? [+m[1], +m[2], +m[3], m[4] !== undefined ? +m[4] : 1] : [200, 200, 220, 0.18];
}
// Blend multiple rgba strings — higher overlap → more opaque
function blendBgs(bgs) {
  if (bgs.length === 1) return bgs[0];
  const vals = bgs.map(parseRgba);
  const r = Math.round(vals.reduce((s, v) => s + v[0], 0) / vals.length);
  const g = Math.round(vals.reduce((s, v) => s + v[1], 0) / vals.length);
  const b = Math.round(vals.reduce((s, v) => s + v[2], 0) / vals.length);
  // opacity scales with sqrt of count so 2 overlaps ~= 1.4× opacity
  const baseA = vals.reduce((s, v) => s + v[3], 0) / vals.length;
  const a = Math.min(0.65, baseA * Math.sqrt(bgs.length));
  return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}

function renderLines(text, rules) {
  const lines = text.split('\n');
  const withPos = rules.filter(r => getRuleLineRange(r));
  if (rules.length && !withPos.length)
    console.warn('renderLines: have', rules.length, 'rules but none have line_start — no highlights');
  const lineMap = new Map();
  for (const r of rules) {
    const range = getRuleLineRange(r);
    if (!range) continue;
    const col = colorFor(r.extracted_by || (r.source === 'hand' ? S.userName : r.source) || '?');
    const ls = range[0] - 1;
    const le = Math.min(range[1] - 1, lines.length - 1);
    for (let i = ls; i <= le; i++) {
      if (!lineMap.has(i)) lineMap.set(i, []);
      lineMap.get(i).push({ r, col });
    }
  }
  return lines.map((line, idx) => {
    const entries = lineMap.get(idx);
    if (!entries || !entries.length) return escHtml(line) + '\n';
    const rids  = entries.map(e => e.r.id).join(',');
    // blend backgrounds; each extractor's color contributes equally
    const bg  = blendBgs(entries.map(e => e.col.bg));
    const hov = blendBgs(entries.map(e => e.col.hov));
    const act = blendBgs(entries.map(e => e.col.act));
    // inset box-shadow: one 3px bar per unique extractor (max 4), no text shift
    const uniq = [...new Map(entries.map(e => [e.col.bdr, e.col])).values()];
    const shadow = uniq.slice(0, 4).map((c, i) => `inset ${(i + 1) * 3}px 0 0 ${c.bdr}`).join(',');
    const title = escAttr(entries.map(e => (e.r.extracted_by || e.r.source || '?') + ': ' + e.r.rule_text.slice(0, 60)).join(' | '));
    return `<span class="line-hl" data-line="${idx+1}" data-rids="${rids}" data-bg="${bg}" data-hov="${hov}" data-act="${act}"` +
      ` style="background:${bg};box-shadow:${shadow}"` +
      ` title="${title}">${escHtml(line)}</span>\n`;
  }).join('');
}

// ─── Text selection ──────────────────────────────────────────
function onViewerMouseUp() {
  const pre = document.getElementById('contentPre');
  if (!pre) return;
  const lineNo = getCollapsedSelectionLineNo(pre);
  if (lineNo != null) {
    clearSelection();
    applyLineSearch(lineNo);
    return;
  }
  const offsets = getSelectionOffsets(pre);
  if (!offsets) { clearSelection(); return; }
  S.selection = offsets;
  const preview = offsets.text.slice(0,55).replace(/\s+/g,' ');
  setSelInfo(`"${preview}${offsets.text.length>55?'…':''}" (${offsets.end-offsets.start} chars) — <kbd>⌘↵</kbd>`);
  document.getElementById('addBtn').disabled = false;
}

function clearSelection() {
  S.selection = null;
  document.getElementById('addBtn').disabled = true;
  setSelInfo(null);
}

function setSelInfo(html) {
  const el = document.getElementById('selInfo');
  el.innerHTML = html || `<span class="kb">Select text then <kbd>⌘↵</kbd> to add &nbsp;·&nbsp;
    <kbd>j</kbd><kbd>k</kbd> nav &nbsp;·&nbsp; <kbd>n</kbd> note &nbsp;·&nbsp; <kbd>d</kbd> del</span>`;
}

function boundaryOffset(container, node, offset) {
  const r = document.createRange();
  r.selectNodeContents(container);
  r.setEnd(node, offset);
  return r.toString().length;
}

function getCollapsedSelectionLineNo(container) {
  const sel = window.getSelection();
  if (!sel || !sel.isCollapsed || !sel.rangeCount) return null;
  const range = sel.getRangeAt(0);
  if (!container.contains(range.startContainer)) return null;
  const start = boundaryOffset(container, range.startContainer, range.startOffset);
  if (!Number.isFinite(start) || !S.currentFile?.content) return null;
  return S.currentFile.content.slice(0, start).split('\n').length;
}

// Robust offset using Range boundaries – handles mixed text/spans correctly
function getSelectionOffsets(container) {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
  const range = sel.getRangeAt(0);
  if (!container.contains(range.commonAncestorContainer)) return null;
  const text = sel.toString();
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
  });
  if (saved.error) { setStatus(saved.error,'error'); return; }

  S.rules.push(saved);
  clearSelection();
  window.getSelection()?.removeAllRanges();
  renderViewer();
  renderRulesPanel();
  refreshFileBadge(S.currentFile.id);
  setFocusedRule(saved.id);
  setTimeout(() => openNoteEditor(saved.id), 60);
}

// ─── Rules panel ─────────────────────────────────────────────
function renderRulesPanel() {
  const visible = getVisibleRules();
  const hasSearch = !!S.ruleSearchQuery.trim();
  const searchLabel = hasSearch ? ` matching "${esc(S.ruleSearchQuery.trim())}"` : '';
  const visibleIds = new Set(visible.map(r => r.id));
  S.selectedIds = new Set([...S.selectedIds].filter(id => visibleIds.has(id)));
  if (S.focusedRuleId && !visibleIds.has(S.focusedRuleId)) setFocusedRule(null);
  if (S.lastSelectedId && !visibleIds.has(S.lastSelectedId)) S.lastSelectedId = null;
  const hand = visible.filter(r => r.source === 'hand');
  const llm  = visible.filter(r => r.source === 'llm');
  document.getElementById('handCount').textContent = hand.length;
  document.getElementById('llmCount').textContent  = llm.length;
  document.getElementById('handRules').innerHTML = hand.length
    ? hand.map(r => ruleCard(r)).join('')
    : (hasSearch
        ? `<div class="empty-hint">No hand rules${searchLabel}</div>`
        : '<div class="empty-hint">Select text to add rules</div>');
  document.getElementById('llmRules').innerHTML = llm.length
    ? llm.map(r => ruleCard(r)).join('')
    : (hasSearch
        ? `<div class="empty-hint">No LLM rules${searchLabel}</div>`
        : '<div class="empty-hint">Run the LLM to extract rules</div>');
  document.getElementById('clearRuleSearchBtn').disabled = !S.ruleSearchQuery.trim();
  document.getElementById('rulesVisibleCount').textContent = `${visible.length}/${S.rules.length}`;
  updateBulkBar();
  updateExtractClearButton();
}

function promoteRuleToSectionTop(id) {
  const idx = S.rules.findIndex(r => r.id === id);
  if (idx < 0) return false;
  const src = S.rules[idx].source;
  const firstInSection = S.rules.findIndex(r => r.source === src);
  if (firstInSection < 0 || idx === firstInSection) return false;
  const [rule] = S.rules.splice(idx, 1);
  S.rules.splice(firstInSection, 0, rule);
  return true;
}

function focusAndPromoteRule(id) {
  if (promoteRuleToSectionTop(id)) renderRulesPanel();
  focusRule(id);
}

function selectedLlmIds() {
  const selected = new Set(S.selectedIds);
  return S.rules
    .filter(r => r.source === 'llm' && selected.has(r.id))
    .map(r => r.id);
}

function focusedLlmId() {
  const focused = S.rules.find(r => r.id === S.focusedRuleId);
  return focused?.source === 'llm' ? focused.id : null;
}

function updateExtractClearButton() {
  const btn = document.getElementById('clearExtractBtn');
  if (!btn) return;
  const selected = selectedLlmIds();
  if (selected.length) {
    btn.disabled = false;
    btn.textContent = `Clear selected (${selected.length})`;
    return;
  }
  const focusedId = focusedLlmId();
  if (focusedId) {
    btn.disabled = false;
    btn.textContent = 'Clear focused';
    return;
  }
  btn.disabled = !S.lastExtractRunId;
  btn.textContent = 'Clear last run';
}

function ruleCard(r) {
  const extractor = r.extracted_by || (r.source==='hand' ? S.userName : r.source) || '?';
  const col    = colorFor(extractor);
  const hasPos = r.char_start != null || r.line_start != null;
  const pos    = r.line_start ? `L${r.line_start}${r.line_end&&r.line_end!==r.line_start?'–'+r.line_end:''}` : 'no position';
  const preview = r.rule_text.slice(0,110).replace(/\s+/g,' ');
  const focused = S.focusedRuleId === r.id ? ' focused' : '';
  const hasNote = r.notes ? ' has-note' : '';
  const sel = S.selectedIds.has(r.id) ? ' selected' : '';
  return `<div class="rule-card${focused}${sel}${hasPos?'':' no-pos'}" id="rc-${r.id}"
               style="${rcVars(col)}" onclick="ruleCardClick(event,'${r.id}')">
    <label class="rule-checkbox" onclick="event.stopPropagation()">
      <input type="checkbox" ${S.selectedIds.has(r.id)?'checked':''} onchange="toggleRuleSelect('${r.id}',this.checked,event)">
    </label>
    <div class="rule-body">
      <div><span class="extractor-tag" style="${rcVars(col)}">${esc(extractor)}</span></div>
      <div class="rule-preview">${esc(preview)}${r.rule_text.length>110?'…':''}</div>
      <div class="rule-pos">${pos}</div>
      ${r.notes ? `<div class="rule-note-text">${esc(r.notes)}</div>` : ''}
      <div class="rule-note-editor" id="rne-${r.id}">
        <textarea class="rule-note-input" id="rni-${r.id}" rows="2"
          placeholder="Add a note… (Enter save, Esc cancel)"
          onkeydown="noteKeyDown(event,'${r.id}')"
        >${esc(r.notes||'')}</textarea>
        <div class="note-hint"><kbd>↵</kbd> save &nbsp; <kbd>Esc</kbd> cancel</div>
      </div>
    </div>
    <div class="rule-actions">
      <button class="rule-btn note-btn${hasNote}" onclick="toggleNote(event,'${r.id}')" title="Note (n)">✎</button>
      <button class="rule-btn del" onclick="deleteRule(event,'${r.id}')" title="Delete (d)">×</button>
    </div>
  </div>`;
}

// ─── Focus / navigation ──────────────────────────────────────
function setFocusedRule(id) {
  // Restore previous focused line highlights
  if (S.focusedRuleId) {
    document.querySelectorAll(`.line-hl[data-rids]`).forEach(el => {
      if (el.dataset.rids.split(',').includes(S.focusedRuleId))
        el.style.background = el.dataset.bg;
    });
  }
  S.focusedRuleId = id;
  document.querySelectorAll('.rule-card').forEach(el => el.classList.remove('focused'));
  if (id) {
    const card = document.getElementById('rc-'+id);
    card?.classList.add('focused');
    card?.scrollIntoView({ behavior:'smooth', block:'nearest' });
    document.querySelectorAll(`.line-hl[data-rids]`).forEach(el => {
      if (el.dataset.rids.split(',').includes(id))
        el.style.background = el.dataset.act;
    });
  }
  updateExtractClearButton();
}

function focusRule(id) {
  setFocusedRule(id);
  const first = [...document.querySelectorAll('.line-hl[data-rids]')]
    .find(el => el.dataset.rids.split(',').includes(id));
  if (first) first.scrollIntoView({ behavior:'smooth', block:'center' });
}

function allRuleCards() { return [...document.querySelectorAll('.rule-card')]; }

function moveFocus(delta) {
  const cards = allRuleCards();
  if (!cards.length) return;
  const idx  = cards.findIndex(c => c.id === 'rc-'+S.focusedRuleId);
  const next = cards[Math.max(0, Math.min(cards.length-1, idx+delta))];
  if (next) focusRule(next.id.replace('rc-',''));
}

// ─── Notes ───────────────────────────────────────────────────
function toggleNote(evt, id) {
  evt.stopPropagation();
  S.editingNoteId === id ? closeNoteEditor(id, false) : openNoteEditor(id);
}
function openNoteEditor(id) {
  if (S.editingNoteId && S.editingNoteId !== id) closeNoteEditor(S.editingNoteId, false);
  setFocusedRule(id); S.editingNoteId = id;
  const ed = document.getElementById('rne-'+id);
  const inp = document.getElementById('rni-'+id);
  if (!ed||!inp) return;
  ed.classList.add('open'); inp.focus();
  inp.setSelectionRange(inp.value.length, inp.value.length);
}
function closeNoteEditor(id, save) {
  const ed = document.getElementById('rne-'+id);
  const inp = document.getElementById('rni-'+id);
  if (!ed) return;
  if (save && inp) saveNote(id, inp.value);
  ed.classList.remove('open'); S.editingNoteId = null;
}
async function saveNote(id, text) {
  const notes = text.trim() || null;
  await api(`/api/rules/${id}`, 'PATCH', { notes });
  const rule = S.rules.find(r => r.id === id);
  if (rule) rule.notes = notes;
  const card = document.getElementById('rc-'+id);
  if (card) {
    const wasOpen = S.editingNoteId === id;
    const prevFocused = S.focusedRuleId;
    card.outerHTML = ruleCard(rule);
    S.focusedRuleId = prevFocused;
    document.getElementById('rc-'+id)?.classList.toggle('focused', prevFocused === id);
    if (wasOpen) S.editingNoteId = null;
  }
}
function noteKeyDown(evt, id) {
  if (evt.key === 'Enter' && !evt.shiftKey) { evt.preventDefault(); closeNoteEditor(id, true); }
  if (evt.key === 'Escape') { evt.preventDefault(); closeNoteEditor(id, false); }
  evt.stopPropagation();
}

// ─── Bulk selection ──────────────────────────────────────────
function ruleCardClick(evt, id) {
  if (evt.shiftKey && S.lastSelectedId) {
    rangeSelect(S.lastSelectedId, id);
    return;
  }
  focusAndPromoteRule(id);
  S.lastSelectedId = id;
}

function rangeSelect(fromId, toId) {
  const ids = allRuleCards().map(c => c.id.replace('rc-', ''));
  const a = ids.indexOf(fromId), b = ids.indexOf(toId);
  if (a === -1 || b === -1) return;
  const [lo, hi] = a < b ? [a, b] : [b, a];
  ids.slice(lo, hi + 1).forEach(rid => S.selectedIds.add(rid));
  S.lastSelectedId = toId;
  renderRulesPanel();
}

function toggleRuleSelect(id, checked, evt) {
  if (evt?.shiftKey && S.lastSelectedId && checked) {
    rangeSelect(S.lastSelectedId, id);
    return;
  }
  checked ? S.selectedIds.add(id) : S.selectedIds.delete(id);
  document.getElementById('rc-'+id)?.classList.toggle('selected', checked);
  if (checked) S.lastSelectedId = id;
  updateBulkBar();
}

function selectAll(source) {
  getVisibleRules().filter(r => r.source === source).forEach(r => S.selectedIds.add(r.id));
  renderRulesPanel();
}

function clearRuleSelection() {
  S.selectedIds.clear();
  renderRulesPanel();
}

function updateBulkBar() {
  const n = S.selectedIds.size;
  const bar = document.getElementById('bulkBar');
  bar.classList.toggle('visible', n > 0);
  document.getElementById('bulkCount').textContent = `${n} selected`;
  updateExtractClearButton();
}

async function bulkDelete() {
  if (!S.selectedIds.size) return;
  const ids = [...S.selectedIds];
  for (const id of ids) await api(`/api/rules/${id}`, 'DELETE');
  S.rules = S.rules.filter(r => !ids.includes(r.id));
  if (ids.includes(S.focusedRuleId)) S.focusedRuleId = null;
  S.selectedIds.clear();
  renderViewer(); renderRulesPanel();
  refreshFileBadge(S.currentFile?.id);
}

async function bulkMerge() {
  if (S.selectedIds.size < 2) return;
  const ids = [...S.selectedIds];
  const selected = S.rules.filter(r => ids.includes(r.id))
    .sort((a, b) => (a.line_start||0) - (b.line_start||0));
  const line_start = Math.min(...selected.map(r => r.line_start || 0).filter(Boolean));
  const line_end   = Math.max(...selected.map(r => r.line_end   || 0).filter(Boolean));
  const rule_text  = selected.map(r => r.rule_text).join('\n');
  const saved = await api('/api/rules', 'POST', {
    file_id: S.currentFile.id, rule_text,
    char_start: null, char_end: null,
    line_start: line_start || null, line_end: line_end || null,
    extracted_by: S.userName || 'merged',
  });
  if (saved.error) { alert(saved.error); return; }
  for (const id of ids) await api(`/api/rules/${id}`, 'DELETE');
  S.rules = S.rules.filter(r => !ids.includes(r.id));
  S.rules.push(saved);
  S.selectedIds.clear();
  renderViewer(); renderRulesPanel();
  refreshFileBadge(S.currentFile?.id);
  setFocusedRule(saved.id);
}

// ─── Delete ──────────────────────────────────────────────────
async function deleteRule(evt, id) {
  evt.stopPropagation();
  const cards = allRuleCards();
  const idx   = cards.findIndex(c => c.id === 'rc-'+id);
  const nextId = (cards[idx+1]||cards[idx-1])?.id.replace('rc-','') || null;
  await api(`/api/rules/${id}`, 'DELETE');
  S.rules = S.rules.filter(r => r.id !== id);
  S.selectedIds.delete(id);
  if (S.focusedRuleId === id) S.focusedRuleId = nextId;
  if (S.editingNoteId === id) S.editingNoteId = null;
  renderViewer(); renderRulesPanel();
  if (nextId) setFocusedRule(nextId);
  refreshFileBadge(S.currentFile?.id);
}

// ─── Collapsible sub-sections ────────────────────────────────
function toggleSub(bodyId, arrowId) {
  const body  = document.getElementById(bodyId);
  const arrow = document.getElementById(arrowId);
  const collapsed = body.classList.toggle('collapsed');
  arrow?.classList.toggle('closed', collapsed);
}

// ─── Extraction ──────────────────────────────────────────────
async function runExtraction() {
  if (!S.currentFile) return;
  const prompt = document.getElementById('extractPrompt').value.trim();
  const model  = document.getElementById('extractModel').value;
  if (!prompt) { setExtractStatus('Enter a prompt', 'error'); return; }
  document.getElementById('extractBtn').disabled = true;
  setExtractStatus('Running…');
  const res = await api('/api/llm', 'POST', { file_id: S.currentFile.id, prompt, model });
  document.getElementById('extractBtn').disabled = false;
  if (res.error) { setExtractStatus(res.error, 'error'); return; }
  const rules = res.rules;
  if (!Array.isArray(rules)) {
    console.error('runExtraction: unexpected response', res);
    setExtractStatus('Unexpected response — check console', 'error'); return;
  }
  console.log('runExtraction: received', rules.length, 'rules');
  console.log('  with line_start:', rules.filter(r => r.line_start != null).length);
  console.log('  sample:', rules.slice(0,3).map(r => ({id:r.id, line_start:r.line_start, text:r.rule_text?.slice(0,40)})));
  S.lastExtractRunId = res.run_id;
  const existing = new Set(S.rules.map(r => r.id));
  for (const r of rules) if (!existing.has(r.id)) S.rules.push(r);
  const matched = rules.filter(r => r.line_start != null).length;
  setExtractStatus(`${rules.length} rules extracted, ${matched} located`, matched > 0 ? 'ok' : 'error');
  renderViewer(); renderRulesPanel();
  refreshFileBadge(S.currentFile.id);
}

async function clearExtraction() {
  const selected = selectedLlmIds();
  if (selected.length) {
    for (const id of selected) await api(`/api/rules/${id}`, 'DELETE');
    S.rules = S.rules.filter(r => !(r.source === 'llm' && selected.includes(r.id)));
    selected.forEach(id => S.selectedIds.delete(id));
    if (selected.includes(S.focusedRuleId)) S.focusedRuleId = null;
    setExtractStatus(`Cleared ${selected.length} selected rule${selected.length===1?'':'s'}`, 'ok');
    renderViewer(); renderRulesPanel();
    refreshFileBadge(S.currentFile?.id);
    return;
  }
  const focusedId = focusedLlmId();
  if (focusedId) {
    await api(`/api/rules/${focusedId}`, 'DELETE');
    S.rules = S.rules.filter(r => r.id !== focusedId);
    S.selectedIds.delete(focusedId);
    if (S.focusedRuleId === focusedId) S.focusedRuleId = null;
    setExtractStatus('Cleared focused rule', 'ok');
    renderViewer(); renderRulesPanel();
    refreshFileBadge(S.currentFile?.id);
    return;
  }
  if (!S.lastExtractRunId) return;
  await api(`/api/llm-runs/${S.lastExtractRunId}`, 'DELETE');
  S.rules = S.rules.filter(r => r.llm_run_id !== S.lastExtractRunId);
  S.lastExtractRunId = null;
  setExtractStatus('Cleared last run', 'ok');
  renderViewer(); renderRulesPanel();
  refreshFileBadge(S.currentFile?.id);
}

function setExtractStatus(msg, cls='') {
  const el = document.getElementById('extractStatus');
  el.textContent = msg; el.className = 'llm-status' + (cls ? ' ' + cls : '');
}

// ─── Classification LLM ──────────────────────────────────────
function updateRunBtn() {
  // button label is fixed; no-op kept for compat
}

async function runLLM() {
  if (!S.currentFile) return;
  const prompt = document.getElementById('llmPrompt').value.trim();
  const model  = document.getElementById('llmModel').value;
  if (!prompt) { setStatus('Enter a prompt', 'error'); return; }
  if (!S.focusedRuleId) { setStatus('Focus a rule first', 'error'); return; }

  document.getElementById('runBtn').disabled = true;
  setStatus('Running…');
  const res = await api('/api/llm', 'POST', {
    file_id: S.currentFile.id, prompt, model, rule_id: S.focusedRuleId
  });
  document.getElementById('runBtn').disabled = false;
  if (res.error) { setStatus(res.error, 'error'); return; }

  S.lastRunId = res.run_id;
  document.getElementById('clearRunBtn').disabled = false;
  showClassification(res.classification, res.raw_response);
  setStatus('Classification complete', 'ok');
}

function showClassification(c, rawJson) {
  const el = document.getElementById('classifResult');
  if (!c || !Object.keys(c).length) {
    el.style.display = 'none'; return;
  }
  el.style.display = 'block';

  const enfClass = c.deterministic_enforceability?.classification || '';
  const ambLevel = c.ambiguity?.level || '';
  const conf     = c.confidence || '';

  const enfColors = {
    deterministically_enforceable_as_written: 'green',
    deterministically_enforceable_with_context: 'blue',
    deterministically_enforceable_after_rewrite: 'blue',
    partially_enforceable: 'yellow',
    requires_llm_or_human_judgment: 'orange',
    not_enforceable: 'red',
  };
  const ambColors = { none:'green', low:'blue', medium:'yellow', high:'orange', not_actionable:'red' };
  const confColors = { high:'green', medium:'yellow', low:'orange' };

  const badge = (text, cls) => `<span class="cr-badge ${cls||'gray'}">${esc(text)}</span>`;
  const row = (label, html) => `<div class="cr-row"><span class="cr-label">${label}</span><span class="cr-val">${html}</span></div>`;

  const prereqs = (c.prerequisites||[]).map(p =>
    `<b>${esc(p.type||'')}</b>: ${esc(p.details||'')}`).join('<br>');
  const mech = c.enforcement_mechanism
    ? `${badge(c.enforcement_mechanism.primary||'','gray')} ${esc(c.enforcement_mechanism.details||'')}` : '';
  const triggers = (c.trigger_location?.recommended||[]).map(t => badge(t,'blue')).join(' ');
  const rewrites = [...(c.suggested_enforceable_rewrite?.deterministic_core||[]),
                   ...(c.suggested_enforceable_rewrite?.semantic_or_review_residual||[])]
    .map((r,i) => `${i+1}. ${esc(r)}`).join('<br>');
  const ambTerms = (c.ambiguity?.ambiguous_terms||[]).join(', ');

  el.innerHTML =
    row('Summary',   esc(c.summary||''))                                        +
    row('Enforceability', badge(enfClass, enfColors[enfClass]||'gray') + ' ' + esc(c.deterministic_enforceability?.details||'')) +
    row('Ambiguity', badge(ambLevel, ambColors[ambLevel]||'gray') + (ambTerms ? ` — ${esc(ambTerms)}` : '')) +
    row('Mechanism', mech)                                                       +
    row('Triggers',  triggers + ' ' + esc(c.trigger_location?.details||''))    +
    (prereqs ? row('Prerequisites', prereqs) : '')                               +
    (rewrites ? row('Rewrite', rewrites) : '')                                   +
    row('Confidence', badge(conf, confColors[conf]||'gray'))                    +
    `<button class="cr-toggle" onclick="this.nextSibling.classList.toggle('open')">▸ Raw JSON</button>` +
    `<div class="cr-json"><pre>${esc(JSON.stringify(c, null, 2))}</pre></div>`;
}

async function clearLastRun() {
  if (!S.lastRunId) return;
  await api(`/api/llm-runs/${S.lastRunId}`, 'DELETE');
  S.rules = S.rules.filter(r => r.llm_run_id !== S.lastRunId);
  S.lastRunId = null;
  document.getElementById('clearRunBtn').disabled = true;
  setStatus('Last run cleared');
  renderViewer(); renderRulesPanel();
  refreshFileBadge(S.currentFile?.id);
}

function setStatus(msg, cls='') {
  const el = document.getElementById('llmStatus');
  el.textContent = msg; el.className = 'llm-status'+(cls?' '+cls:'');
}

// ─── Prompt history ──────────────────────────────────────────
function getPromptHistory() {
  try { return JSON.parse(localStorage.getItem('promptHistory')||'[]'); }
  catch { return []; }
}

async function savePrompt() {
  const text = document.getElementById('llmPrompt').value.trim();
  if (!text) return;
  // Save to server
  const res = await api('/api/prompts', 'POST', { prompt: text });
  if (res.error) { setStatus(res.error,'error'); return; }
  // Also cache locally
  const history = getPromptHistory();
  history.unshift({ id: res.id, text, saved_at: res.saved_at || new Date().toISOString() });
  localStorage.setItem('promptHistory', JSON.stringify(history.slice(0,30)));
  renderPromptHistoryPanel(history);
  setStatus('Prompt saved','ok');
}

function toggleHistory() {
  const el = document.getElementById('promptHistory');
  el.classList.toggle('open');
  if (el.classList.contains('open')) loadPromptHistory();
}

async function loadPromptHistory() {
  // Try server first
  const res = await api('/api/prompts');
  if (Array.isArray(res)) {
    const history = res.map(r => ({ id:r.id, text:r.prompt, label:r.label, saved_at:r.saved_at }));
    localStorage.setItem('promptHistory', JSON.stringify(history));
    renderPromptHistoryPanel(history);
  } else {
    renderPromptHistoryPanel(getPromptHistory());
  }
}

function renderPromptHistoryPanel(history) {
  const el = document.getElementById('promptHistory');
  if (!history.length) {
    el.innerHTML = '<div class="ph-item" style="color:#aaa">No saved prompts</div>';
    return;
  }
  el.innerHTML = history.map(h => {
    const d = h.saved_at ? new Date(h.saved_at).toLocaleDateString() : '';
    const label = h.label ? `<span class="ph-label">${esc(h.label)}</span>` : '';
    return `<div class="ph-item">
      ${label}
      <span class="ph-text" onclick="loadPrompt(${JSON.stringify(h.text)})"
            title="${escAttr(h.text)}">${esc(h.text.slice(0,70))}${h.text.length>70?'…':''}</span>
      <span class="ph-date">${d}</span>
      <button class="ph-del" onclick="deletePrompt(event,'${h.id}')" title="Delete">×</button>
    </div>`;
  }).join('');
}

function loadPrompt(text) {
  document.getElementById('llmPrompt').value = text;
  document.getElementById('promptHistory').classList.remove('open');
}

async function deletePrompt(evt, id) {
  evt.stopPropagation();
  await api(`/api/prompts/${id}`, 'DELETE');
  await loadPromptHistory();
}

// ─── Keyboard shortcuts ──────────────────────────────────────
document.addEventListener('keydown', e => {
  const tag = document.activeElement?.tagName?.toLowerCase();
  const inText = tag==='textarea'||tag==='input';

  // ⌘↵ – add rule (skip if in LLM prompt textarea)
  if ((e.metaKey||e.ctrlKey) && e.key==='Enter') {
    if (document.activeElement?.id === 'llmPrompt') return;
    if (S.selection) { e.preventDefault(); addHandRule(); }
    return;
  }
  if (inText) return;

  switch (e.key) {
    case 'j': case 'ArrowDown': e.preventDefault(); moveFocus(+1); break;
    case 'k': case 'ArrowUp':   e.preventDefault(); moveFocus(-1); break;
    case 'n':
      e.preventDefault();
      if (S.focusedRuleId) openNoteEditor(S.focusedRuleId);
      break;
    case 'd':
      e.preventDefault();
      if (S.focusedRuleId) deleteRule({ stopPropagation(){} }, S.focusedRuleId);
      break;
    case 'x':
      e.preventDefault();
      if (S.focusedRuleId) {
        if (e.shiftKey && S.lastSelectedId) { rangeSelect(S.lastSelectedId, S.focusedRuleId); break; }
        toggleRuleSelect(S.focusedRuleId, !S.selectedIds.has(S.focusedRuleId));
      }
      break;
    case 'Escape':
      if (S.editingNoteId) closeNoteEditor(S.editingNoteId, false);
      else clearSelection();
      break;
  }
});

// Hover rule card → highlight mark
const rulesPanel = document.querySelector('.rules-panel');
rulesPanel.addEventListener('mouseover', e => {
  const card = e.target.closest('.rule-card');
  if (!card || card.id==='rc-'+S.focusedRuleId) return;
  document.querySelector(`mark.hl[data-rid="${card.id.replace('rc-','')}"]`)?.classList.add('hover');
});
rulesPanel.addEventListener('mouseout', e => {
  const card = e.target.closest('.rule-card');
  if (!card) return;
  document.querySelector(`mark.hl[data-rid="${card.id.replace('rc-','')}"]`)?.classList.remove('hover');
});

// ─── Export ──────────────────────────────────────────────────
function exportCSV() {
  window.open('/export', '_blank');
}

// ─── Helpers ─────────────────────────────────────────────────
async function api(url, method='GET', body=null) {
  try {
    const opts = {method, headers:{}};
    if (body) { opts.headers['Content-Type']='application/json'; opts.body=JSON.stringify(body); }
    const r = await fetch(url, opts); return r.json();
  } catch(e) { return {error:e.message}; }
}
function refreshFileBadge(id) {
  if (!id) return;
  const f = S.allFiles.find(f=>f.id===id);
  if (f) {
    f.hand_count = S.rules.filter(r=>r.source==='hand').length;
    f.llm_count  = S.rules.filter(r=>r.source==='llm').length;
  }
  filterFiles(document.getElementById('fileSearch').value);
}
function esc(s) {
  if (s==null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escAttr(s) { return s.replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;'); }
function fmtSize(n) { return n>1024?`${(n/1024).toFixed(1)}k`:`${n||'?'}b`; }

init();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    print(f"Rule Annotator → http://{args.host}:{args.port}")
    app.run(debug=True, host=args.host, port=args.port, threaded=False)
