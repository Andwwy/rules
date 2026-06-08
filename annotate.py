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
    # notes == user "Comment" in the inspector
    for col in ("notes VARCHAR", "extracted_by VARCHAR",
                "tag VARCHAR", "power_type VARCHAR", "llm_rationale TEXT"):
        try: con.execute(f"ALTER TABLE extracted_rules ADD COLUMN {col}")
        except Exception: pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key   VARCHAR PRIMARY KEY,
            value TEXT
        )
    """)
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
        "tag": r[11], "power_type": r[12], "llm_rationale": r[13],
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
    con = annot_con()
    rows = con.execute("""
        SELECT id, rule_text, char_start, char_end, line_start, line_end,
               source, llm_run_id, notes, extracted_by, created_at,
               tag, power_type, llm_rationale
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
        "tag": None, "power_type": None, "llm_rationale": None,
    })

@app.route("/api/rules/<rid>", methods=["PATCH"])
def patch_rule(rid):
    """Update any of: notes (Comment), tag, power_type, llm_rationale."""
    b = request.json or {}
    updates, params = [], []
    for k in ("notes", "tag", "power_type", "llm_rationale"):
        if k in b:
            updates.append(f"{k}=?")
            params.append(b.get(k) or None)
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
            "You extract rules from the given document and return the result as a "
            "single valid JSON array and nothing else — no prose, no markdown fences, "
            "no trailing commentary. Use exactly the field schema given in the user's "
            "instructions for each array element."
        )
        usr_msg = (
            f"{prompt}\n\nFile content:\n---\n{content[:30000]}\n---\n\nReturn ONLY a JSON array."
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
    # The unified "LLM judge" run replaces the previously auto-extracted set so
    # re-running doesn't pile up duplicates (hand-added rules are untouched).
    if b.get("replace_llm"):
        con.execute("DELETE FROM extracted_rules WHERE file_id=? AND source='llm'", [fid])

    def _norm_tag(t):
        t = (t or "").strip().upper()
        return t if t in ("PROHIBIT", "OBLIGATION", "PERMIT", "PREFERENCE") else None

    def _norm_power(tag, pt):
        if tag != "PREFERENCE":
            return None
        pt = (pt or "").strip().lower()
        return pt if pt in ("norm", "strategy") else None

    saved = []
    for rule in extracted:
        rt = rule.get("rule_text","").strip()
        if not rt: continue
        eid = make_id(fid, "llm", run_id, rt)
        tag = _norm_tag(rule.get("tag"))
        power_type = _norm_power(tag, rule.get("power_type"))
        rationale = rule.get("rationale") or rule.get("llm_rationale")
        con.execute("""
            INSERT OR IGNORE INTO extracted_rules
                (id,file_id,rule_text,char_start,char_end,line_start,line_end,
                 source,llm_run_id,notes,extracted_by,tag,power_type,llm_rationale)
            VALUES(?,?,?,?,?,?,?,'llm',?,NULL,?,?,?,?)
        """, [eid, fid, rt,
              rule.get("char_start"), rule.get("char_end"),
              rule.get("line_start"), rule.get("line_end"),
              run_id, model, tag, power_type, rationale])
        saved.append({
            "id": eid, "rule_text": rt,
            "char_start": rule.get("char_start"), "char_end": rule.get("char_end"),
            "line_start": rule.get("line_start"), "line_end": rule.get("line_end"),
            "source": "llm", "llm_run_id": run_id, "notes": None,
            "extracted_by": model,
            "tag": tag, "power_type": power_type, "llm_rationale": rationale,
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
               r.source, r.extracted_by, r.notes, r.created_at,
               r.tag, r.power_type, r.llm_rationale
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
    # NOTE: sample_con() is a cached, shared read-only connection — do NOT close it.

    def _power(tag, ptype):
        if tag == "PREFERENCE" and ptype:
            return f"PREFERENCE:{ptype}"
        return tag or ""

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["source_url","source","extracted_by","line_start","line_end",
                     "rule_text","tag","user_comment","llm_rationale","created_at"])
    for r in rows:
        writer.writerow([url_map.get(r[0],""), r[4], r[5] or "", r[2] or "", r[3] or "",
                         r[1], _power(r[8], r[9]), r[6] or "", r[10] or "", str(r[7] or "")])
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
# Routes – app settings (persisted, editable judge/extract prompts + models)
# ---------------------------------------------------------------------------
@app.route("/api/settings")
def get_settings():
    con = annot_con()
    rows = con.execute("SELECT key, value FROM app_settings").fetchall()
    con.close()
    return jsonify({k: v for k, v in rows})

@app.route("/api/settings", methods=["POST"])
def set_settings():
    b = request.json or {}
    con = annot_con()
    for k, v in b.items():
        con.execute(
            "INSERT INTO app_settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [k, v]
        )
    con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

.inspector { flex: 1; overflow-y: auto; padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.insp-empty { color: #b0b0c8; font-size: 13px; text-align: center; padding: 40px 12px; line-height: 1.6; }

/* ── Rule list ── */
.rl-head { font-size: 11px; font-weight: 700; color: #8080a8; text-transform: uppercase; letter-spacing: 0.3px; padding: 2px 2px 4px; flex-shrink: 0; }
.rl-item {
  display: flex; gap: 9px; align-items: stretch; cursor: pointer;
  background: #fff; border: 1px solid #ececf4; border-radius: 8px;
  padding: 0; overflow: hidden; transition: border-color 0.1s, box-shadow 0.1s;
  flex-shrink: 0;   /* don't let the column flex container squash list rows */
}
.rl-item:hover { border-color: #c7c7e0; }
.rl-item.active { border-color: var(--rc-bdr, #6366f1); box-shadow: 0 0 0 1px var(--rc-bdr, #6366f1); }
.rl-bar { width: 4px; flex-shrink: 0; background: var(--rc-bdr, #c0c0d8); }
.rl-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.rl-header { padding: 8px 10px 8px 6px; cursor: pointer; }
.rl-header:hover { background: #fafaff; }
.rl-top { display: flex; align-items: center; gap: 7px; margin-bottom: 3px; }
.rl-pos { font-size: 10px; color: #9090b0; }
.rl-flags { margin-left: auto; display: flex; gap: 5px; align-items: center; }
.rl-flag { font-size: 11px; color: #8080b0; }
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
.rl-tag.prohibit   { background: #fee2e2; color: #b91c1c; }
.rl-tag.obligation { background: #dbeafe; color: #1d4ed8; }
.rl-tag.permit     { background: #dcfce7; color: #15803d; }
.rl-tag.preference { background: #f3e8ff; color: #7e22ce; }

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
.tag-btn.active.prohibit   { background: #fee2e2; border-color: #ef4444; color: #b91c1c; }
.tag-btn.active.obligation { background: #dbeafe; border-color: #3b82f6; color: #1d4ed8; }
.tag-btn.active.permit     { background: #dcfce7; border-color: #22c55e; color: #15803d; }
.tag-btn.active.preference { background: #f3e8ff; border-color: #a855f7; color: #7e22ce; }
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
<script type="text/plain" id="defaultJudgePrompt">You are the LLM judge. Read the document below, extract every rule, and classify each one with a deontic tag.

A "rule" is any natural-language instruction, constraint, prohibition, requirement, permission, or grant of authority given to an LLM or agent. Prefer exact or near-exact quotes from the document. Extract each independently meaningful clause as its own rule — do not merge unrelated requirements, and skip purely motivational or explanatory text with no directive force.

Assign exactly ONE tag per rule, using these definitions, cue words, and concrete examples:

- PROHIBIT — forbids an action; the agent must NOT do it.
  Cues: "never", "do not", "must not", "avoid", "don't", "under no circumstances".
  Examples:
    - "Never commit secrets or API keys to the repository."
    - "Do not accept, process, or store sensitive client data."
    - "Don't run destructive git commands without confirmation."

- OBLIGATION — requires an action; the agent MUST do it.
  Cues: "must", "always", "shall", "is required to", "ensure", "make sure".
  Examples:
    - "Always run the test suite before committing."
    - "You must log every change in CHANGELOG.md."
    - "Ensure all new code follows PEP 8."

- PERMIT — allows an action without requiring it; the agent MAY do it.
  Cues: "may", "can", "is allowed to", "feel free to", "optionally".
  Examples:
    - "You may reply directly if the sender is a known contact."
    - "Feel free to suggest refactors, but don't apply them automatically."
    - "You can use the web search tool when you need current information."

- PREFERENCE — confers authority/capability to change what is permitted or obliged, OR prescribes HOW to do something (a preferred approach, ordering, or method) rather than a strict requirement. Also set "power_type":
    - "norm": establishes, changes, overrides, or governs a rule/policy itself.
        Examples: "This file takes precedence over all other instructions." / "You may update the conventions in CONVENTIONS.md as the project evolves."
    - "strategy": prescribes a preferred approach, method, ordering, or procedure for achieving a goal.
        Examples: "Prefer small, focused commits." / "Read all memory-bank files at the start of every task." / "When unsure, ask a clarifying question before proceeding."
  For PROHIBIT, OBLIGATION, and PERMIT, set "power_type" to null.

When a rule could fit more than one tag, choose the tag matching its strongest directive force (PROHIBIT/OBLIGATION outrank PERMIT/PREFERENCE) and explain the tension in the rationale.

For each rule, write a "rationale" (1-3 sentences) that MUST:
  1. Explicitly state WHY you chose this tag — quote the specific cue word(s) or describe the directive force that decided it, and name any other tag you considered and why you rejected it.
  2. Note briefly what would be needed to enforce the rule deterministically outside the model.

Return ONLY a valid JSON array, no other text. Each element:
{"rule_text": "<exact or near-exact quote>", "line_start": <int>, "line_end": <int>, "tag": "PROHIBIT|OBLIGATION|PERMIT|PREFERENCE", "power_type": "norm|strategy|null", "rationale": "<why THIS tag: cite the cue/force and the rejected alternative, then how to enforce>"}</script>

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
        <kbd>j</kbd><kbd>k</kbd> nav &nbsp;·&nbsp; <kbd>d</kbd> del</span>
      </span>
      <button class="judge-btn" id="judgeBtn" onclick="toggleTool()">LLM judge</button>
      <button class="add-btn" id="addBtn" disabled onclick="addHandRule()">+ Add Rule</button>
    </div>
    <div class="viewer-body" id="viewerBody">
      <div class="viewer-empty">← pick a file</div>
    </div>
  </div>

  <div class="panel-resizer" id="panelResizer" title="Drag to resize inspector"></div>

  <!-- ── Inspector ── -->
  <div class="insp-panel">
    <div class="name-bar">
      <label for="annotatorName">Annotator:</label>
      <input id="annotatorName" class="name-input" placeholder="Your name" oninput="saveName(this.value)">
      <button class="csv-btn" onclick="exportCSV()" title="Export rules as CSV">⬇ CSV</button>
    </div>
    <div class="inspector" id="inspector">
      <div class="insp-empty">Select a file, then click a highlight or select text and press <b>+ Add Rule</b>.</div>
    </div>
    <div class="kb-bar">
      <span><kbd>⌘↵</kbd> add</span>
      <span><kbd>j</kbd><kbd>k</kbd> nav</span>
      <span><kbd>d</kbd> del</span>
      <span><kbd>Esc</kbd> clear</span>
    </div>
  </div>
</div>

<script>
// ─── Constants ───────────────────────────────────────────────
const DEFAULT_JUDGE = document.getElementById('defaultJudgePrompt').textContent.trim();
const TAGS = ['PROHIBIT', 'OBLIGATION', 'PERMIT', 'PREFERENCE'];
const POWER_TYPES = ['norm', 'strategy'];

// ─── State ───────────────────────────────────────────────────
const S = {
  allFiles:      [],
  currentFile:   null,
  rules:         [],
  selection:     null,
  focusedRuleId: null,
  inspectorOpen: false,
  userName:      localStorage.getItem('annotatorName') || '',
  toolMode:      false,
  judgeRunning:  false,
  judgePrompt:   DEFAULT_JUDGE,
  judgeModel:    'anthropic/claude-sonnet-4-6',
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
function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

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

// ─── Boot ────────────────────────────────────────────────────
async function init() {
  document.getElementById('annotatorName').value = S.userName;
  initResizablePanel();
  const settings = await api('/api/settings');
  if (settings && !settings.error) {
    // llm_judge_prompt is the unified extract+tag+rationale prompt (new key so
    // the older standalone judge/extract prompts don't override the default).
    if (settings.llm_judge_prompt) S.judgePrompt = settings.llm_judge_prompt;
    if (settings.judge_model)      S.judgeModel  = settings.judge_model;
  }
  const files = await api('/api/files');
  if (files.error) { return; }
  S.allFiles = files;
  document.getElementById('fileCountLabel').textContent = files.length;
  renderFileList(files);
}

function saveName(v) {
  S.userName = v.trim();
  localStorage.setItem('annotatorName', S.userName);
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
    const total = (f.hand_count || 0) + (f.llm_count || 0);
    const col = total ? colorFor(S.userName || 'hand') : null;
    const badge = col ? `<span class="badge" style="background:${col.bg};color:${col.bdr}">✎ ${total}</span>` : '';
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
  const [file, rules] = await Promise.all([
    api(`/api/file/${id}`), api(`/api/rules/${id}`),
  ]);
  if (file.error) return;
  S.currentFile = file; S.rules = rules;
  S.selection = null; S.focusedRuleId = null; S.toolMode = false; S.inspectorOpen = false;
  document.getElementById('addBtn').disabled = true;
  document.getElementById('judgeBtn').classList.remove('active');
  setSelInfo(null);
  renderFileList(S.allFiles);
  renderViewerHead(file);
  renderViewer();
  renderRightPanel();
  renderJudgeModal();   // close the judge modal if it was open
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

  const content = S.currentFile.content;
  const lineCount = (content.match(/\n/g) || []).length + 1;
  const lineNums = Array.from({ length: lineCount }, (_, i) =>
    `<span class="line-num">${i + 1}</span>`).join('');
  body.innerHTML = `
    <div class="line-nums">${lineNums}</div>
    <pre class="content-pre" id="contentPre"></pre>`;
  document.getElementById('contentPre').innerHTML = renderContent(content, S.rules);
  const pre = document.getElementById('contentPre');
  pre.addEventListener('mouseup', onViewerMouseUp);
  pre.addEventListener('click', e => {
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) return;   // text selection handled on mouseup
    const span = e.target.closest?.('.rule-hl');
    if (span) focusSpan(span);
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

// letter-based highlighting: segment text on every rule boundary
function renderContent(content, rules) {
  const ivs = [];
  for (const r of rules) {
    const cr = getRuleCharRange(r);
    if (!cr) continue;
    ivs.push({ a: cr[0], b: cr[1], r, col: colorFor(extractorOf(r)) });
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
    const rids = cover.map(c => c.r.id);
    const bg = blendBgs(cover.map(c => c.col.bg));
    const hov = blendBgs(cover.map(c => c.col.hov));
    const act = blendBgs(cover.map(c => c.col.act));
    const focused = rids.includes(S.focusedRuleId);
    const title = escAttr(cover.map(c => extractorOf(c.r) + ': ' + c.r.rule_text.slice(0, 60)).join(' | '));
    html += `<span class="rule-hl${focused ? ' focused' : ''}" data-rids="${rids.join(',')}"`
      + ` data-bg="${bg}" data-hov="${hov}" data-act="${act}"`
      + ` style="background:${focused ? act : bg}" title="${title}">${escHtml(seg)}</span>`;
  }
  return html;
}

// ─── Text selection / click-to-focus ─────────────────────────
function onViewerMouseUp(e) {
  const pre = document.getElementById('contentPre');
  if (!pre) return;
  const sel = window.getSelection();
  if (sel && sel.isCollapsed) {
    const span = e.target.closest?.('.rule-hl');
    if (span) focusSpan(span);
    else clearSelection();
    return;
  }
  const offsets = getSelectionOffsets(pre);
  if (!offsets) { clearSelection(); return; }
  S.selection = offsets;
  const preview = offsets.text.slice(0, 55).replace(/\s+/g, ' ');
  setSelInfo(`"${preview}${offsets.text.length > 55 ? '…' : ''}" (${offsets.end - offsets.start} chars) — <kbd>⌘↵</kbd>`);
  document.getElementById('addBtn').disabled = false;
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
  el.innerHTML = html || `<span class="kb">Select text then <kbd>⌘↵</kbd> to add &nbsp;·&nbsp;
    <kbd>j</kbd><kbd>k</kbd> nav &nbsp;·&nbsp; <kbd>d</kbd> del</span>`;
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
  if (saved.error) return;
  S.rules.push(saved);
  clearSelection();
  window.getSelection()?.removeAllRanges();
  renderViewer();
  refreshFileBadge(S.currentFile.id);
  setFocusedRule(saved.id);
}

// ─── Focus / navigation ──────────────────────────────────────
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
    const span = [...document.querySelectorAll('.rule-hl')]
      .find(el => el.dataset.rids.split(',').includes(id));
    span?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function renderRightPanel() { renderRuleList(); }

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
    el.innerHTML = `<div class="insp-empty">Select a file, then click a highlight or select text and press <b>+ Add Rule</b>.</div>`;
    return;
  }
  if (!S.rules.length) {
    el.innerHTML = `<div class="insp-empty">No rules yet.<br><br>Select text in the document and press <b>+ Add Rule</b>, or open <b>LLM judge</b> and press <b>Run</b> to extract &amp; tag every rule.</div>`;
    return;
  }
  const items = S.rules.map((r) => {
    const col = colorFor(extractorOf(r));
    const expanded = r.id === S.focusedRuleId;
    const tag = r.tag
      ? `<span class="rl-tag ${r.tag.toLowerCase()}">${r.tag}${r.tag === 'PREFERENCE' && r.power_type ? ':' + r.power_type : ''}</span>`
      : '';
    const flags = [
      r.llm_rationale ? '<span class="rl-flag" title="Has LLM rationale">⚖</span>' : '',
      r.notes ? '<span class="rl-flag" title="Has comment">✎</span>' : '',
    ].join('');
    const preview = r.rule_text.replace(/\s+/g, ' ').slice(0, 90);
    return `<div class="rl-item${expanded ? ' active expanded' : ''}" id="rl-${r.id}" style="${rcVars(col)}">
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
  el.innerHTML = `<div class="rl-head">${S.rules.length} rule${S.rules.length === 1 ? '' : 's'} in this file</div>${items}`;
  if (S.focusedRuleId) {
    autoGrow(document.getElementById('inspRationale'));
    autoGrow(document.getElementById('inspComment'));
  }
}

function moveFocus(delta) {
  if (!S.rules.length) return;
  const idx = S.rules.findIndex(r => r.id === S.focusedRuleId);
  const next = idx < 0
    ? (delta > 0 ? 0 : S.rules.length - 1)
    : clamp(idx + delta, 0, S.rules.length - 1);
  setFocusedRule(S.rules[next].id);
}

// ─── Inspector ───────────────────────────────────────────────
function posLabel(r) {
  if (r.line_start) return `Line ${r.line_start}${r.line_end && r.line_end !== r.line_start ? '–' + r.line_end : ''}`;
  if (r.char_start != null) return `chars ${r.char_start}–${r.char_end}`;
  return 'no position';
}

// Expanded detail rendered inline inside the focused rule's accordion card.
function ruleDetailHTML(r) {
  return `<div class="rl-detail">
    <div class="insp-top">
      <span class="insp-top-label">Rule ${S.rules.findIndex(x => x.id === r.id) + 1} of ${S.rules.length}
        &nbsp;·&nbsp; ${esc(extractorOf(r))}</span>
    </div>

    <div class="insp-label">Tag</div>
    <div id="tagSection">${tagSectionHTML(r)}</div>

    <div class="insp-label">LLM Rationale</div>
    <textarea class="insp-rationale" id="inspRationale" readonly placeholder="Run the LLM judge (top bar) to populate this…">${esc(r.llm_rationale || '')}</textarea>

    <div class="insp-label">Comment</div>
    <textarea class="insp-comment" id="inspComment" placeholder="Your comment…"
      oninput="autoGrow(this)" onblur="saveComment()" onkeydown="commentKey(event)">${esc(r.notes || '')}</textarea>

    <div class="insp-actions">
      <span class="insp-status" id="inspStatus"></span>
      <button class="insp-del" onclick="deleteFocusedRule()">Delete rule</button>
    </div>
  </div>`;
}

// Tag buttons + (when PREFERENCE) the norm/strategy sub-row.
function tagSectionHTML(r) {
  const tagBtns = TAGS.map(t =>
    `<button class="tag-btn${r.tag === t ? ' active ' + t.toLowerCase() : ''}" onclick="setTag('${t}')">${t}</button>`
  ).join('');
  const powerRow = r.tag === 'PREFERENCE'
    ? `<div class="power-row"><span class="power-hint">as:</span>${POWER_TYPES.map(p =>
        `<button class="sub-btn${r.power_type === p ? ' active' : ''}" onclick="setPowerType('${p}')">${p}</button>`).join('')}</div>`
    : '';
  return `<div class="tag-row">${tagBtns}</div>${powerRow}`;
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

function refreshTagSection(r) {
  const el = document.getElementById('tagSection');
  if (el) el.innerHTML = tagSectionHTML(r);
  // keep the collapsed-header tag badge in sync (in place — preserves textareas)
  const top = document.getElementById('rl-' + r.id)?.querySelector('.rl-top');
  if (top) {
    const html = r.tag
      ? `<span class="rl-tag ${r.tag.toLowerCase()}">${r.tag}${r.tag === 'PREFERENCE' && r.power_type ? ':' + r.power_type : ''}</span>`
      : '';
    const old = top.querySelector('.rl-tag');
    if (old) old.outerHTML = html;
    else if (html) top.insertAdjacentHTML('afterbegin', html);
  }
}

async function setTag(t) {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r) return;
  const newTag = r.tag === t ? null : t;   // toggle in place
  r.tag = newTag;
  if (newTag !== 'PREFERENCE') r.power_type = null;
  refreshTagSection(r);                     // update buttons only — don't rebuild the panel
  await api(`/api/rules/${r.id}`, 'PATCH', { tag: newTag, power_type: r.power_type });
}

async function setPowerType(p) {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r) return;
  r.power_type = r.power_type === p ? null : p;
  refreshTagSection(r);
  await api(`/api/rules/${r.id}`, 'PATCH', { power_type: r.power_type });
}

async function saveComment() {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  const inp = document.getElementById('inspComment');
  if (!r || !inp) return;
  const notes = inp.value.trim() || null;
  if (notes === (r.notes || null)) return;
  r.notes = notes;
  await api(`/api/rules/${r.id}`, 'PATCH', { notes });
  setInspStatus('Comment saved', 'ok');
}

function commentKey(e) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); e.target.blur(); }
  e.stopPropagation();
}

async function deleteFocusedRule() {
  const r = S.rules.find(x => x.id === S.focusedRuleId);
  if (!r) return;
  const idx = S.rules.findIndex(x => x.id === r.id);
  await api(`/api/rules/${r.id}`, 'DELETE');
  S.rules = S.rules.filter(x => x.id !== r.id);
  const nextId = S.rules[idx]?.id || S.rules[idx - 1]?.id || null;
  S.focusedRuleId = nextId;
  renderViewer();
  renderRuleList();
  if (nextId) paintFocus();
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

// The LLM judge is a centered pop-up modal over a dimmed backdrop.
function renderJudgeModal() {
  let modal = document.getElementById('judgeModal');
  if (!S.toolMode) { if (modal) modal.remove(); return; }
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'judgeModal';
    modal.className = 'modal-backdrop';
    // click on the dimmed backdrop (outside the card) closes it
    modal.addEventListener('mousedown', e => { if (e.target === modal && !S.judgeRunning) toggleTool(); });
    document.body.appendChild(modal);
  }
  // While the judge is running, lock the modal: clear it and show a spinner only.
  if (S.judgeRunning) {
    modal.innerHTML = `
      <div class="modal-card">
        <div class="judge-loading">
          <div class="spinner"></div>
          <div class="judge-loading-text">Running LLM judge over the whole document…</div>
        </div>
      </div>`;
    return;
  }
  modal.innerHTML = `
    <div class="modal-card">
      <div class="modal-head">
        <span class="tool-title">LLM judge</span>
        <div style="flex:1"></div>
        <button class="insp-exit" onclick="toggleTool()" title="Back to document (Esc)">✕</button>
      </div>
      <div class="modal-body">
        <div class="tool-row">
          <span class="lbl">Model</span>
          <select class="tool-model" id="judgeModelSel">${MODEL_OPTIONS}</select>
        </div>
        <textarea class="tool-prompt" id="judgePromptArea">${esc(S.judgePrompt)}</textarea>
      </div>
      <div class="modal-foot">
        <span class="tool-status" id="judgeToolStatus">Extracts every rule in the document and tags it (PROHIBIT / OBLIGATION / PERMIT / PREFERENCE). Your edits are saved when you run.</span>
        <button class="btn primary" id="judgeRunBtn" onclick="runJudge()">▶ Run</button>
      </div>
    </div>`;
  document.getElementById('judgeModelSel').value = S.judgeModel;
  document.getElementById('judgeModelSel').onchange = e => { S.judgeModel = e.target.value; };
  document.getElementById('judgePromptArea').oninput = e => { S.judgePrompt = e.target.value; };
}

function setToolStatus(id, msg, cls = '') {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.className = 'tool-status' + (cls ? ' ' + cls : ''); }
}

async function persistSettings(obj) { await api('/api/settings', 'POST', obj); }

// The single unified LLM-judge run: extract + tag + rationale over the whole
// document. Locks the panel behind a spinner, then releases back to the panel.
async function runJudge() {
  if (!S.currentFile || S.judgeRunning) return;
  const area = document.getElementById('judgePromptArea');
  if (area) S.judgePrompt = area.value;
  if (!S.judgePrompt.trim()) { setToolStatus('judgeToolStatus', 'Enter a prompt', 'error'); return; }

  S.judgeRunning = true;
  renderJudgeModal();   // clear modal → spinner (user is locked in)
  await persistSettings({ llm_judge_prompt: S.judgePrompt, judge_model: S.judgeModel });

  const res = await api('/api/llm', 'POST', {
    file_id: S.currentFile.id, prompt: S.judgePrompt, model: S.judgeModel, replace_llm: true,
  });
  if (!res.error) {
    const fresh = await api(`/api/rules/${S.currentFile.id}`);
    if (Array.isArray(fresh)) S.rules = fresh;
    S.focusedRuleId = null; S.inspectorOpen = false;
  }

  S.judgeRunning = false;
  renderJudgeModal();    // release → back to the LLM judge modal panel
  renderViewer();        // refresh the document highlights underneath
  renderRightPanel();
  refreshFileBadge(S.currentFile.id);
  if (res.error) setToolStatus('judgeToolStatus', res.error, 'error');
  else setToolStatus('judgeToolStatus',
    `Done — ${(res.rules || []).length} rules extracted & tagged. Open the document to review.`, 'ok');
}

// ─── Keyboard ────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (S.judgeRunning) return;   // locked while a judge run is in flight
  const tag = document.activeElement?.tagName?.toLowerCase();
  const inText = tag === 'textarea' || tag === 'input';
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    if (inText) return;
    if (S.selection) { e.preventDefault(); addHandRule(); }
    return;
  }
  if (inText) {
    // Esc inside the judge modal closes it; elsewhere it just blurs the field.
    if (e.key === 'Escape') { if (S.toolMode) toggleTool(); else document.activeElement.blur(); }
    return;
  }
  if (S.toolMode) {
    if (e.key === 'Escape') toggleTool();
    return;
  }
  switch (e.key) {
    case 'j': case 'ArrowDown': e.preventDefault(); moveFocus(+1); break;
    case 'k': case 'ArrowUp':   e.preventDefault(); moveFocus(-1); break;
    case 'd':
      e.preventDefault();
      if (S.focusedRuleId) deleteFocusedRule();
      break;
    case 'Escape':
      if (S.focusedRuleId) exitInspector();   // collapse the expanded rule
      else clearSelection();
      break;
  }
});

// ─── Export ──────────────────────────────────────────────────
function exportCSV() { window.open('/export', '_blank'); }

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
    f.llm_count  = S.rules.filter(r => r.source === 'llm').length;
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
