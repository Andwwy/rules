#!/usr/bin/env python3
"""
app_v2.py - V2 rules browser with marking and reclassification actions.

This is a dedicated UI for processed_v2.db so the workflow is explicit:
  - browse V2 rules
  - mark rules you do not like
  - filter to marked / low-confidence rules
  - rerun reclassification from the UI
"""

from __future__ import annotations

import json
import os
from typing import Optional

import duckdb
import openai
from flask import Flask, jsonify, request

import process_v2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_V2_DB = os.path.join(BASE_DIR, "processed_v2.db")
DEFAULT_MODEL = process_v2.DEFAULT_MODEL

app = Flask(__name__)
_ro_cons: dict[str, duckdb.DuckDBPyConnection] = {}


def _ro(path: str):
    if path not in _ro_cons:
        if not os.path.exists(path):
            return None
        try:
            _ro_cons[path] = duckdb.connect(path, read_only=True)
        except Exception as exc:
            print(f"[warn] cannot open {path}: {exc}")
            return None
    return _ro_cons[path]


def close_ro(path: str):
    con = _ro_cons.pop(path, None)
    if con is not None:
        try:
            con.close()
        except Exception:
            pass


def proc_con():
    return _ro(PROCESSED_V2_DB)


def proc_rw():
    try:
        close_ro(PROCESSED_V2_DB)
        con = duckdb.connect(PROCESSED_V2_DB, read_only=False)
        process_v2.init_out_db(con)
        return con
    except Exception as exc:
        print(f"[warn] cannot open processed_v2.db for write: {exc}")
        return None


def refresh_ro():
    close_ro(PROCESSED_V2_DB)


def parse_examples(raw_examples: str) -> list[dict]:
    raw_examples = (raw_examples or "").strip()
    if not raw_examples:
        return []
    data = json.loads(raw_examples)
    if not isinstance(data, list):
        raise ValueError("examples must be a JSON array")
    return [item for item in data if isinstance(item, dict)]


def build_where(
    search: str,
    rule_type: str,
    enforcement_mode: str,
    source: str,
    marked_only: bool,
    confidence_below: Optional[float],
):
    wheres = []
    params = []

    if search:
        wheres.append("LOWER(rule_text) LIKE ?")
        params.append(f"%{search.lower()}%")
    if rule_type and rule_type != "all":
        wheres.append("rule_type = ?")
        params.append(rule_type)
    if enforcement_mode and enforcement_mode != "all":
        wheres.append("enforcement_mode = ?")
        params.append(enforcement_mode)
    if source:
        wheres.append("rule_source = ?")
        params.append(source)
    if marked_only:
        wheres.append("manual_reprocess = TRUE")
    if confidence_below is not None:
        wheres.append("(confidence IS NULL OR confidence < ?)")
        params.append(confidence_below)

    where = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    return where, params


@app.route("/api/summary")
def api_summary():
    con = proc_con()
    if not con:
        return jsonify({"error": "processed_v2.db not found"}), 404

    total = con.execute("SELECT COUNT(*) FROM processed_rules_v2").fetchone()[0]
    marked = con.execute(
        "SELECT COUNT(*) FROM processed_rules_v2 WHERE manual_reprocess = TRUE"
    ).fetchone()[0]
    low = con.execute(
        "SELECT COUNT(*) FROM processed_rules_v2 WHERE confidence IS NULL OR confidence < 0.7"
    ).fetchone()[0]
    types = con.execute(
        """
        SELECT COALESCE(rule_type, '(none)') AS k, COUNT(*) AS n
        FROM processed_rules_v2
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall()
    modes = con.execute(
        """
        SELECT COALESCE(enforcement_mode, '(none)') AS k, COUNT(*) AS n
        FROM processed_rules_v2
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall()
    sources = con.execute(
        """
        SELECT COALESCE(rule_source, '(none)') AS k, COUNT(*) AS n
        FROM processed_rules_v2
        GROUP BY 1 ORDER BY 2 DESC LIMIT 50
        """
    ).fetchall()

    return jsonify({
        "total": total,
        "marked": marked,
        "low_confidence_default": low,
        "rule_types": [{"key": r[0], "count": r[1]} for r in types],
        "enforcement_modes": [{"key": r[0], "count": r[1]} for r in modes],
        "sources": [{"key": r[0], "count": r[1]} for r in sources],
    })


@app.route("/api/rules")
def api_rules():
    con = proc_con()
    if not con:
        return jsonify({"error": "processed_v2.db not found"}), 404

    search = (request.args.get("search") or "").strip()
    rule_type = (request.args.get("rule_type") or "all").strip()
    enforcement_mode = (request.args.get("enforcement_mode") or "all").strip()
    source = (request.args.get("source") or "").strip()
    marked_only = request.args.get("marked_only", "").lower() in {"1", "true", "yes"}
    confidence_raw = (request.args.get("confidence_below") or "").strip()
    confidence_below = float(confidence_raw) if confidence_raw else None
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)

    where, params = build_where(
        search, rule_type, enforcement_mode, source, marked_only, confidence_below
    )

    total = con.execute(
        f"SELECT COUNT(*) FROM processed_rules_v2 {where}",
        params,
    ).fetchone()[0]
    rows = con.execute(
        f"""
        SELECT id, source_db, source_id, source_url, repo_name, rule_source, rule_text,
               rule_index, rule_type, enforcement_mode, mechanism, labels_json,
               confidence, reasoning, manual_reprocess, updated_at
        FROM processed_rules_v2 {where}
        ORDER BY manual_reprocess DESC, COALESCE(confidence, 0.0) ASC, source_db, rule_index ASC NULLS LAST, id
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    ).fetchall()

    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "rules": [
            {
                "id": r[0],
                "source_db": r[1],
                "source_id": r[2],
                "source_url": r[3],
                "repo_name": r[4],
                "rule_source": r[5],
                "rule_text": r[6],
                "rule_index": r[7],
                "rule_type": r[8],
                "enforcement_mode": r[9],
                "mechanism": r[10],
                "labels_json": r[11],
                "confidence": r[12],
                "reasoning": r[13],
                "manual_reprocess": bool(r[14]),
                "updated_at": str(r[15]) if r[15] is not None else None,
            }
            for r in rows
        ],
    })


@app.route("/api/rules/<rule_id>/mark", methods=["PATCH"])
def api_mark_rule(rule_id):
    con = proc_rw()
    if not con:
        return jsonify({"error": "processed_v2.db not available"}), 503

    body = request.json or {}
    marked = bool(body.get("marked", True))
    try:
        con.execute(
            "UPDATE processed_rules_v2 SET manual_reprocess = ? WHERE id = ?",
            [marked, rule_id],
        )
    finally:
        con.close()
    refresh_ro()
    return jsonify({"ok": True, "marked": marked, "id": rule_id})


@app.route("/api/reclassify", methods=["POST"])
def api_reclassify():
    body = request.json or {}
    confidence_raw = body.get("confidence_below")
    confidence_below = None if confidence_raw in (None, "", False) else float(confidence_raw)
    include_marked = bool(body.get("include_marked", True))
    limit = int(body.get("limit", 0) or 0)
    workers = int(body.get("workers", 5) or 5)
    model = (body.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    instructions = (body.get("instructions") or "").strip()

    try:
        examples = parse_examples(body.get("examples_json") or "")
    except Exception as exc:
        return jsonify({"error": f"Invalid examples JSON: {exc}"}), 400

    selected_ids = body.get("rule_ids") or []
    if not isinstance(selected_ids, list):
        return jsonify({"error": "rule_ids must be a JSON array"}), 400
    selected_ids = [str(value).strip() for value in selected_ids if str(value).strip()]

    con = proc_rw()
    if not con:
        return jsonify({"error": "processed_v2.db not available"}), 503

    try:
        api_key = process_v2.ensure_api_key()
        client = openai.OpenAI(api_key=api_key)
        prompt = process_v2.build_reclassify_prompt(instructions, examples)
        result = process_v2.reclassify_rules(
            con,
            client,
            model,
            prompt,
            confidence_below,
            selected_ids,
            include_marked=include_marked,
            limit=limit,
            workers=workers,
        )
    except Exception as exc:
        con.close()
        refresh_ro()
        return jsonify({"error": str(exc)}), 500

    con.close()
    refresh_ro()
    return jsonify({
        "ok": True,
        "reclassified": result["ok"],
        "errors": result["err"],
    })


@app.route("/")
def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rules V2</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #eef3f8;
  --panel: #0f172a;
  --card: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --line: #d9e2ec;
  --accent: #0f766e;
  --danger: #b91c1c;
  --warn: #c2410c;
  --green: #15803d;
  --amber: #b45309;
  --pink: #be185d;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: radial-gradient(circle at top left, #f7fbff, #eef3f8 45%, #e7eef6);
  color: var(--ink);
  height: 100vh;
  overflow: hidden;
}
.layout { display: flex; height: 100vh; }
.sidebar {
  width: 320px;
  background: rgba(15,23,42,0.96);
  color: white;
  display: flex;
  flex-direction: column;
}
.sidebar h1 {
  padding: 16px 18px;
  font-size: 15px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sidebar-inner {
  padding: 14px 16px 22px;
  overflow-y: auto;
}
.section-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #738196;
  margin: 16px 0 6px;
}
.summary-box, .action-box {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  border-radius: 12px;
  padding: 12px;
}
.summary-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 13px;
  color: #dbe3ef;
  padding: 3px 0;
}
.input, .select, .textarea {
  width: 100%;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  color: white;
  border-radius: 10px;
  padding: 9px 10px;
  font-size: 13px;
}
.textarea { min-height: 96px; resize: vertical; }
.input::placeholder, .textarea::placeholder { color: #7a879b; }
.inline-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.checkbox-row {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #dbe3ef;
  font-size: 13px;
  margin-top: 10px;
}
.btn {
  border: 0;
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 13px;
  cursor: pointer;
}
.btn.primary {
  background: linear-gradient(135deg, #0f766e, #0f9a8c);
  color: white;
}
.btn.ghost {
  background: rgba(255,255,255,0.06);
  color: white;
  border: 1px solid rgba(255,255,255,0.08);
}
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.main-header {
  background: rgba(255,255,255,0.82);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--line);
  padding: 14px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.main-scroll {
  overflow-y: auto;
  padding: 18px 20px 40px;
}
.header-title { font-size: 14px; color: #475569; }
.header-note { font-size: 12px; color: #64748b; }
.rules-list {
  max-width: 980px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.card {
  background: rgba(255,255,255,0.94);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 8px 28px rgba(15,23,42,0.05);
}
.rule-text {
  font-size: 15px;
  line-height: 1.6;
  white-space: pre-wrap;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 12px;
}
.tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  font-size: 11px;
  border-radius: 999px;
  background: #eff6ff;
  color: #1d4ed8;
}
.tag.det { background: #ecfdf5; color: var(--green); }
.tag.mix { background: #fffbeb; color: var(--amber); }
.tag.llm { background: #fdf2f8; color: var(--pink); }
.tag.low { background: #fff7ed; color: var(--warn); }
.tag.marked { background: #fee2e2; color: var(--danger); }
.reasoning {
  margin-top: 12px;
  color: #475569;
  font-size: 13px;
  line-height: 1.55;
}
.actions {
  margin-top: 14px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.mark-btn {
  border: 1px solid #d9e2ec;
  background: white;
  color: #334155;
  border-radius: 999px;
  padding: 7px 11px;
  font-size: 12px;
  cursor: pointer;
}
.mark-btn.marked {
  border-color: #fecaca;
  background: #fff1f2;
  color: var(--danger);
}
.link {
  font-size: 12px;
  color: #0f766e;
  text-decoration: none;
}
.link:hover { text-decoration: underline; }
.status {
  font-size: 12px;
  color: var(--muted);
  margin-top: 10px;
  min-height: 18px;
}
.error {
  color: var(--danger);
  background: #fff1f2;
  border: 1px solid #fecaca;
  border-radius: 10px;
  padding: 12px;
}
.load-more {
  margin: 18px auto 0;
  display: block;
}
@media (max-width: 980px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; max-height: 48vh; }
}
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h1>Rules V2</h1>
    <div class="sidebar-inner">
      <div class="section-label">Summary</div>
      <div id="summaryBox" class="summary-box"></div>

      <div class="section-label">Browse</div>
      <input id="searchInput" class="input" placeholder="Search rule text">
      <div class="inline-grid" style="margin-top:8px;">
        <select id="typeSelect" class="select"></select>
        <select id="modeSelect" class="select"></select>
      </div>
      <select id="sourceSelect" class="select" style="margin-top:8px;"></select>
      <div class="inline-grid" style="margin-top:8px;">
        <input id="confidenceFilter" class="input" type="number" min="0" max="1" step="0.01" placeholder="Confidence below">
        <button id="reloadBtn" class="btn ghost">Refresh</button>
      </div>
      <label class="checkbox-row">
        <input id="markedOnly" type="checkbox">
        <span>Show marked only</span>
      </label>

      <div class="section-label">Reclassify</div>
      <div class="action-box">
        <div class="inline-grid">
          <input id="reclassifyThreshold" class="input" type="number" min="0" max="1" step="0.01" placeholder="Confidence below">
          <input id="workerCount" class="input" type="number" min="1" step="1" value="5" placeholder="Workers">
        </div>
        <label class="checkbox-row">
          <input id="includeMarked" type="checkbox" checked>
          <span>Include marked rules</span>
        </label>
        <textarea id="instructionsInput" class="textarea" style="margin-top:10px;" placeholder="Extra prompt instructions (optional)"></textarea>
        <textarea id="examplesInput" class="textarea" style="margin-top:10px;" placeholder='Examples JSON array (optional)'></textarea>
        <button id="reclassifyBtn" class="btn primary" style="margin-top:10px; width:100%;">Reclassify Selection</button>
        <div class="status" id="actionStatus"></div>
      </div>
    </div>
  </div>

  <div class="main">
    <div class="main-header">
      <div id="totalInfo" class="header-title">Loading…</div>
      <div class="header-note">Mark rules you dislike, then rerun reclassification from this UI.</div>
    </div>
    <div class="main-scroll">
      <div id="rulesList" class="rules-list"></div>
      <button id="loadMoreBtn" class="btn ghost load-more" style="display:none;">Load more</button>
    </div>
  </div>
</div>

<script>
const S = {
  search: '',
  ruleType: 'all',
  enforcementMode: 'all',
  source: '',
  markedOnly: false,
  confidenceFilter: '',
  offset: 0,
  limit: 100,
  total: 0,
  loading: false
};

function esc(text) {
  return text == null ? '' : String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function parseLabels(raw) {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function buildQuery() {
  const params = new URLSearchParams();
  if (S.search) params.set('search', S.search);
  if (S.ruleType !== 'all') params.set('rule_type', S.ruleType);
  if (S.enforcementMode !== 'all') params.set('enforcement_mode', S.enforcementMode);
  if (S.source) params.set('source', S.source);
  if (S.markedOnly) params.set('marked_only', 'true');
  if (S.confidenceFilter) params.set('confidence_below', S.confidenceFilter);
  params.set('offset', String(S.offset));
  params.set('limit', String(S.limit));
  return params.toString();
}

function tagClass(mode) {
  if (mode === 'deterministic') return 'det';
  if (mode === 'mixed') return 'mix';
  if (mode === 'llm') return 'llm';
  return '';
}

function cardHTML(rule) {
  const labels = parseLabels(rule.labels_json);
  const confidence = rule.confidence == null ? null : Number(rule.confidence);
  return `
    <div class="card" data-id="${esc(rule.id)}">
      <div class="rule-text">${esc(rule.rule_text)}</div>
      <div class="meta">
        ${rule.rule_type ? `<span class="tag">${esc(rule.rule_type)}</span>` : ''}
        ${rule.enforcement_mode ? `<span class="tag ${tagClass(rule.enforcement_mode)}">${esc(rule.enforcement_mode)}</span>` : ''}
        ${rule.mechanism ? `<span class="tag">${esc(rule.mechanism)}</span>` : ''}
        ${confidence != null ? `<span class="tag ${confidence < 0.7 ? 'low' : ''}">confidence ${confidence.toFixed(2)}</span>` : ''}
        ${rule.manual_reprocess ? `<span class="tag marked">marked</span>` : ''}
        ${rule.source_db ? `<span class="tag">${esc(rule.source_db)}</span>` : ''}
        ${rule.rule_source ? `<span class="tag">${esc(rule.rule_source)}</span>` : ''}
        ${rule.repo_name ? `<span class="tag">${esc(rule.repo_name)}</span>` : ''}
        ${labels.map(label => `<span class="tag">${esc(label)}</span>`).join('')}
      </div>
      ${rule.reasoning ? `<div class="reasoning"><strong>Reasoning:</strong> ${esc(rule.reasoning)}</div>` : ''}
      <div class="actions">
        <button class="mark-btn ${rule.manual_reprocess ? 'marked' : ''}" data-mark="${rule.manual_reprocess ? 'clear' : 'set'}" data-id="${esc(rule.id)}">
          ${rule.manual_reprocess ? 'Unmark' : 'Mark for rerun'}
        </button>
        ${rule.source_url ? `<a class="link" href="${esc(rule.source_url)}" target="_blank">source ↗</a>` : ''}
      </div>
    </div>
  `;
}

async function loadSummary() {
  const data = await api('/api/summary');
  document.getElementById('summaryBox').innerHTML = `
    <div class="summary-row"><span>Total rules</span><strong>${Number(data.total).toLocaleString()}</strong></div>
    <div class="summary-row"><span>Marked</span><strong>${Number(data.marked).toLocaleString()}</strong></div>
    <div class="summary-row"><span>Confidence &lt; 0.70</span><strong>${Number(data.low_confidence_default).toLocaleString()}</strong></div>
  `;

  const typeSelect = document.getElementById('typeSelect');
  typeSelect.innerHTML = [
    '<option value="all">All rule types</option>',
    ...data.rule_types.map(item => `<option value="${esc(item.key === '(none)' ? '' : item.key)}">${esc(item.key)} (${Number(item.count).toLocaleString()})</option>`)
  ].join('');

  const modeSelect = document.getElementById('modeSelect');
  modeSelect.innerHTML = [
    '<option value="all">All enforcement</option>',
    ...data.enforcement_modes.map(item => `<option value="${esc(item.key === '(none)' ? '' : item.key)}">${esc(item.key)} (${Number(item.count).toLocaleString()})</option>`)
  ].join('');

  const sourceSelect = document.getElementById('sourceSelect');
  sourceSelect.innerHTML = [
    '<option value="">All sources</option>',
    ...data.sources.map(item => `<option value="${esc(item.key === '(none)' ? '' : item.key)}">${esc(item.key)} (${Number(item.count).toLocaleString()})</option>`)
  ].join('');
}

async function loadRules(reset = true) {
  if (S.loading) return;
  S.loading = true;
  if (reset) S.offset = 0;

  try {
    const data = await api('/api/rules?' + buildQuery());
    S.total = data.total;
    const list = document.getElementById('rulesList');
    if (reset) list.innerHTML = '';
    const html = data.rules.map(cardHTML).join('');
    if (!html && reset) {
      list.innerHTML = '<div class="error">No rules matched the current filters.</div>';
    } else {
      list.insertAdjacentHTML('beforeend', html);
    }
    S.offset += data.rules.length;
    document.getElementById('totalInfo').textContent =
      `${Number(data.total).toLocaleString()} rules · ${S.markedOnly ? 'marked only' : 'browse'} view`;
    document.getElementById('loadMoreBtn').style.display =
      S.offset < data.total ? 'block' : 'none';
    bindRuleActions();
  } catch (error) {
    document.getElementById('rulesList').innerHTML = `<div class="error">${esc(error.message)}</div>`;
  }
  S.loading = false;
}

function bindRuleActions() {
  document.querySelectorAll('.mark-btn').forEach(button => {
    button.onclick = async () => {
      const id = button.dataset.id;
      const marked = button.dataset.mark === 'set';
      try {
        await api(`/api/rules/${encodeURIComponent(id)}/mark`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ marked })
        });
        await loadSummary();
        await loadRules(true);
      } catch (error) {
        document.getElementById('actionStatus').textContent = error.message;
      }
    };
  });
}

async function rerunClassification() {
  const actionStatus = document.getElementById('actionStatus');
  actionStatus.textContent = 'Reclassifying…';
  try {
    const payload = {
      confidence_below: document.getElementById('reclassifyThreshold').value || '',
      include_marked: document.getElementById('includeMarked').checked,
      workers: Number(document.getElementById('workerCount').value || 5),
      instructions: document.getElementById('instructionsInput').value,
      examples_json: document.getElementById('examplesInput').value
    };
    const data = await api('/api/reclassify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    actionStatus.textContent = `Reclassified ${data.reclassified} rule(s) with ${data.errors} error(s).`;
    await loadSummary();
    await loadRules(true);
  } catch (error) {
    actionStatus.textContent = error.message;
  }
}

function bindControls() {
  document.getElementById('searchInput').addEventListener('input', async event => {
    S.search = event.target.value.trim();
    await loadRules(true);
  });
  document.getElementById('typeSelect').addEventListener('change', async event => {
    S.ruleType = event.target.value || 'all';
    await loadRules(true);
  });
  document.getElementById('modeSelect').addEventListener('change', async event => {
    S.enforcementMode = event.target.value || 'all';
    await loadRules(true);
  });
  document.getElementById('sourceSelect').addEventListener('change', async event => {
    S.source = event.target.value || '';
    await loadRules(true);
  });
  document.getElementById('markedOnly').addEventListener('change', async event => {
    S.markedOnly = event.target.checked;
    await loadRules(true);
  });
  document.getElementById('confidenceFilter').addEventListener('input', async event => {
    S.confidenceFilter = event.target.value.trim();
    await loadRules(true);
  });
  document.getElementById('reloadBtn').onclick = async () => {
    await loadSummary();
    await loadRules(true);
  };
  document.getElementById('reclassifyBtn').onclick = rerunClassification;
  document.getElementById('loadMoreBtn').onclick = () => loadRules(false);
}

async function boot() {
  bindControls();
  await loadSummary();
  await loadRules(true);
}

boot().catch(error => {
  document.getElementById('rulesList').innerHTML = `<div class="error">${esc(error.message)}</div>`;
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V2 rules browser")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()

    print(f"Rules V2 running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
