#!/usr/bin/env python3
"""
process_v2.py - Extract and classify rules with iterative in-place reclassification.

V2 keeps a single current classification per atomic rule. The initial run stays
close to process.py v1:
  raw source file -> one LLM call -> JSON array of atomic rules + classifications

Later iteration runs reclassify only a selected subset of existing rules:
  - rules below a confidence threshold
  - rules manually marked for reclassification
  - specific rule IDs passed on the command line

Previous classifications are overwritten in place; this script does not store
historical runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import duckdb
import openai


# Load .env if present
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_PATH):
    for raw_line in open(ENV_PATH):
        raw_line = raw_line.strip()
        if raw_line and not raw_line.startswith("#") and "=" in raw_line:
            key, value = raw_line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DBS = {
    "rules": os.path.join(BASE_DIR, "rules.db"),
    "rules2": os.path.join(BASE_DIR, "rules2.db"),
    "rules3": os.path.join(BASE_DIR, "rules3.db"),
}

DEFAULT_OUT_DB = os.path.join(BASE_DIR, "processed_v2.db")
DEFAULT_MODEL = "gpt-4o-mini"
MAX_CONTENT_CHARS = 12_000
MIN_CONTENT_CHARS = 30

DEFAULT_RULE_TYPES = [
    "output_rule",
    "process_rule",
]

DEFAULT_ENFORCEMENT_MODES = [
    "deterministic",
    "llm",
    "mixed",
]

DEFAULT_MECHANISMS = [
    "formatter",
    "linter",
    "type_checker",
    "regex",
    "schema_validator",
    "test_runner",
    "ci_gate",
    "git_hook",
    "filesystem_check",
    "llm_judge",
    "human_review",
    "other",
]

DEFAULT_LABELS = [
    "output_format",
    "language_constraint",
    "style",
    "branding",
    "json",
    "tool_use",
    "user_interaction",
    "approval_required",
    "planning",
    "testing",
    "git",
    "docs",
    "code_quality",
    "safety",
    "ui",
    "repo_hygiene",
]

BUILTIN_PROMPT_EXAMPLES = [
    {
        "rule_text": "Respond only in Chinese",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "regex",
        "labels": ["language_constraint"],
        "reasoning": "Rule type: output_rule because it constrains the content of one response in isolation. Enforcement mode: deterministic because the presence of Chinese text can usually be checked mechanically without reasoning about a multi-step workflow.",
    },
    {
        "rule_text": "Use JSON format",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "schema_validator",
        "labels": ["output_format", "json"],
        "reasoning": "Rule type: output_rule because it specifies the format of a single response. Enforcement mode: deterministic because JSON structure can usually be validated directly from the output.",
    },
    {
        "rule_text": "No emojis",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "regex",
        "labels": ["style"],
        "reasoning": "Rule type: output_rule because it constrains the content/style of one response. Enforcement mode: deterministic because emojis can typically be detected with a simple mechanical check.",
    },
    {
        "rule_text": "Models MUST follow naming conventions",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "regex",
        "labels": ["naming_convention", "code_quality"],
        "reasoning": "Rule type: output_rule because it constrains the properties of the produced artifact, specifically how names should appear in code or models. Enforcement mode: deterministic because naming patterns can usually be validated mechanically.",
    },
    {
        "rule_text": "Use TanStack React Query for efficient data fetching and caching",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "filesystem_check",
        "labels": ["data_fetching", "dependencies", "code_quality"],
        "reasoning": "Rule type: output_rule because it constrains the implementation choices in the produced codebase, not the agent's step-by-step behavior. Enforcement mode: deterministic because the presence of the required library and imports can usually be checked mechanically.",
    },
    {
        "rule_text": "Files matching patterns in .gitignore are excluded.",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "filesystem_check",
        "labels": ["repo_hygiene"],
        "reasoning": "Rule type: output_rule because it constrains which files belong in the resulting artifact set or repo view. Enforcement mode: deterministic because file patterns from .gitignore can be parsed and checked programmatically.",
    },
    {
        "rule_text": "Create models in my_project/some_domain/some_domain_struct.py.",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "filesystem_check",
        "labels": ["repo_hygiene", "code_quality"],
        "reasoning": "Rule type: output_rule because it constrains where produced code artifacts should live in the repository. Enforcement mode: deterministic because file paths and module locations can be verified mechanically.",
    },
    {
        "rule_text": "Maintain up-to-date README with setup instructions",
        "rule_type": "output_rule",
        "enforcement_mode": "mixed",
        "mechanism": "human_review",
        "labels": ["docs"],
        "reasoning": "Rule type: output_rule because it constrains the state of a produced documentation artifact rather than a workflow sequence. Enforcement mode: mixed because you can verify that a README exists and changed, but whether it is truly up to date still requires contextual judgment.",
    },
    {
        "rule_text": "Use @database/server for worker resolvers, @database/client for client components.",
        "rule_type": "output_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "filesystem_check",
        "labels": ["dependencies", "code_quality"],
        "reasoning": "Rule type: output_rule because it constrains import and module usage in the produced code. Enforcement mode: deterministic because imports and module references can usually be checked mechanically.",
    },
    {
        "rule_text": "Run tests after writing code",
        "rule_type": "process_rule",
        "enforcement_mode": "deterministic",
        "mechanism": "test_runner",
        "labels": ["testing", "tool_use"],
        "reasoning": "Rule type: process_rule because it governs the order of actions across multiple steps rather than one output. Enforcement mode: deterministic because, in a reasonably instrumented runtime, code-write events and test-run events can be logged and checked in sequence.",
    },
    {
        "rule_text": "Ask the user before executing commands",
        "rule_type": "process_rule",
        "enforcement_mode": "mixed",
        "mechanism": "human_review",
        "labels": ["user_interaction", "approval_required", "safety"],
        "reasoning": "Rule type: process_rule because it constrains decision flow over time before a tool action happens. Enforcement mode: mixed because a runtime can often track whether approval happened before command execution, but the scope and sufficiency of that approval may still depend on context.",
    },
    {
        "rule_text": "Break the task into steps before coding",
        "rule_type": "process_rule",
        "enforcement_mode": "llm",
        "mechanism": "llm_judge",
        "labels": ["planning"],
        "reasoning": "Rule type: process_rule because it governs planning and sequencing across turns. Enforcement mode: llm because deciding whether the task was meaningfully broken into steps usually requires semantic judgment rather than a simple trace check.",
    },
]


def now_utc():
    return datetime.now(timezone.utc)


def load_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def load_examples(path: Optional[str]) -> list[dict]:
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("examples file must contain a JSON array")
    out = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def parse_ids(raw_ids: list[str], ids_file: Optional[str]) -> list[str]:
    values = list(raw_ids or [])
    if ids_file:
        with open(ids_file, encoding="utf-8") as fh:
            file_text = fh.read()
        for token in file_text.replace(",", "\n").splitlines():
            token = token.strip()
            if token:
                values.append(token)
    deduped = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return text


def coerce_confidence(value) -> float:
    if value is None:
        return 0.0
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num > 1.0 and num <= 100.0:
        num = num / 100.0
    if num < 0.0:
        return 0.0
    if num > 1.0:
        return 1.0
    return num


def normalize_labels(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(",", "\n").splitlines()
        labels = parts
    elif isinstance(value, list):
        labels = value
    else:
        return []

    out = []
    seen = set()
    for label in labels:
        if label is None:
            continue
        label = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        if not label:
            continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def normalize_text(value) -> str:
    return str(value or "").strip()


def normalize_prediction(pred: dict, include_rule_text: bool) -> dict:
    if not isinstance(pred, dict):
        raise ValueError("prediction must be a JSON object")

    out = {
        "rule_type": normalize_text(pred.get("rule_type")).lower().replace("-", "_").replace(" ", "_") or None,
        "enforcement_mode": normalize_text(pred.get("enforcement_mode")).lower().replace("-", "_").replace(" ", "_") or None,
        "mechanism": normalize_text(pred.get("mechanism")).lower().replace("-", "_") or None,
        "labels": normalize_labels(pred.get("labels"))[:4],
        "confidence": coerce_confidence(pred.get("confidence")),
        "reasoning": normalize_text(pred.get("reasoning")),
    }
    if include_rule_text:
        out["rule_text"] = normalize_text(pred.get("rule_text"))
    return out


def examples_block(examples: list[dict]) -> str:
    if not examples:
        return ""

    lines = ["Examples to imitate when classifying:"]
    for i, example in enumerate(examples, 1):
        rule_text = normalize_text(example.get("rule_text"))
        if not rule_text:
            continue
        lines.append(f"{i}. rule_text: {rule_text}")
        if example.get("rule_type"):
            lines.append(f"   rule_type: {example['rule_type']}")
        if example.get("enforcement_mode"):
            lines.append(f"   enforcement_mode: {example['enforcement_mode']}")
        if example.get("mechanism") is not None:
            lines.append(f"   mechanism: {example['mechanism']}")
        labels = normalize_labels(example.get("labels"))
        if labels:
            lines.append(f"   labels: {', '.join(labels)}")
        if example.get("reasoning"):
            lines.append(f"   why: {normalize_text(example['reasoning'])}")
    return "\n".join(lines).strip()


def combined_examples_block(examples: list[dict]) -> str:
    merged = list(BUILTIN_PROMPT_EXAMPLES)
    if examples:
        merged.extend(examples)
    return examples_block(merged)


def taxonomy_block() -> str:
    return f"""\
Taxonomy:

Rule types:
- output_rule: A rule that constrains the properties of the produced output or artifact.
  This includes not only one response in isolation, but also produced code, files,
  names, imports, documentation, formatting, paths, and repository structure.
  It should be judged mainly from the resulting artifact(s) themselves rather than from
  the agent's broader workflow over time. Typical signals are wording, format, language,
  structure, length, tone, style, naming, file placement, dependencies, or other
  properties of the final produced artifact(s).
- process_rule: A rule that constrains what the agent should do across multiple steps.
  It should be judged mainly from the sequence of actions, decisions, tool calls, approvals,
  retries, or planning behavior over time rather than from one final response alone.

When a rule could be read both ways:
- prefer output_rule if the core requirement is about the shape/content/properties of the produced artifact(s)
- prefer process_rule if the core requirement is about how the agent behaves or what order
  of actions it follows over time
- mention the ambiguity in reasoning when it is close

Enforcement modes:
- deterministic: Can be checked or enforced from reasonably instrumented runtime, tool,
  or trace signals without needing LLM judgment.
- llm: Still requires LLM judgment even if you have normal runtime or trace signals,
  because it depends on intent, tone, context, or quality.
- mixed: Some parts are checkable from runtime or trace signals, but some parts still
  depend on contextual interpretation.

Mechanism:
- Pick the single best enforcement mechanism when one is obvious.
- Allowed starter mechanisms: {", ".join(DEFAULT_MECHANISMS)}
- Use null if there is no clear mechanism.

Labels:
- Add zero to four concise labels about what the rule is about.
- Prefer this starter vocabulary when it fits: {", ".join(DEFAULT_LABELS)}
- A short new snake_case label is allowed if truly needed.

Confidence:
- Return a confidence score between 0.0 and 1.0.
- This is a triage aid, not a claim of objective certainty.

Reasoning:
- Return one or two explicit sentences.
- Say why the rule is an output_rule or process_rule.
- Say why the enforcement mode is deterministic, llm, or mixed.
- If the case is ambiguous, mention the ambiguity briefly.
"""


def build_extract_prompt(extra_instructions: str, examples: list[dict]) -> str:
    parts = [
        "You are a precise analyst of AI-agent rules.",
        "You will receive the raw text of a file that may contain many rules.",
        "Split the file into individual, self-contained actionable rules.",
        "Discard headings, examples, and prose that are not themselves rules.",
        taxonomy_block(),
        "Return ONLY a JSON array. Each element must be:",
        """{
  "rule_text": "<the extracted atomic rule>",
  "rule_type": "output_rule" | "process_rule",
  "enforcement_mode": "deterministic" | "llm" | "mixed",
  "mechanism": "<string or null>",
  "labels": ["<label1>", "<label2>"],
  "confidence": 0.0,
  "reasoning": "<explicit explanation of rule_type and enforcement_mode>"
}""",
        "Use lowercase snake_case for rule_type, enforcement_mode, mechanism, and labels.",
        "Keep rule_text close to the source wording, with only light cleanup.",
    ]
    if extra_instructions:
        parts.append("Additional instructions:\n" + extra_instructions.strip())
    example_text = combined_examples_block(examples)
    if example_text:
        parts.append(example_text)
    return "\n\n".join(parts)


def build_reclassify_prompt(extra_instructions: str, examples: list[dict]) -> str:
    parts = [
        "You are a precise analyst of AI-agent rules.",
        "You will receive exactly one already-extracted atomic rule.",
        "Do not rewrite the rule. Only classify it.",
        taxonomy_block(),
        "Return ONLY a single JSON object shaped like:",
        """{
  "rule_type": "output_rule" | "process_rule",
  "enforcement_mode": "deterministic" | "llm" | "mixed",
  "mechanism": "<string or null>",
  "labels": ["<label1>", "<label2>"],
  "confidence": 0.0,
  "reasoning": "<explicit explanation of rule_type and enforcement_mode>"
}""",
        "Use lowercase snake_case for rule_type, enforcement_mode, mechanism, and labels.",
    ]
    if extra_instructions:
        parts.append("Additional instructions:\n" + extra_instructions.strip())
    example_text = combined_examples_block(examples)
    if example_text:
        parts.append(example_text)
    return "\n\n".join(parts)


def init_out_db(con: duckdb.DuckDBPyConnection):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_rules_v2 (
            id                   VARCHAR PRIMARY KEY,
            source_db            VARCHAR NOT NULL,
            source_id            VARCHAR NOT NULL,
            source_url           VARCHAR,
            repo_name            VARCHAR,
            rule_source          VARCHAR,
            rule_text            TEXT NOT NULL,
            rule_index           INTEGER,
            rule_type            VARCHAR,
            enforcement_mode     VARCHAR,
            mechanism            VARCHAR,
            labels_json          TEXT,
            confidence           DOUBLE,
            reasoning            TEXT,
            manual_reprocess     BOOLEAN DEFAULT FALSE,
            extraction_model     VARCHAR,
            classification_model VARCHAR,
            updated_at           TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS process_log_v2 (
            source_db    VARCHAR NOT NULL,
            source_id    VARCHAR NOT NULL,
            rules_found  INTEGER,
            status       VARCHAR,
            error        VARCHAR,
            processed_at TIMESTAMP,
            PRIMARY KEY (source_db, source_id)
        )
        """
    )


def already_processed(out_con, source_db: str, source_id: str) -> bool:
    row = out_con.execute(
        "SELECT 1 FROM process_log_v2 WHERE source_db = ? AND source_id = ?",
        [source_db, source_id],
    ).fetchone()
    return row is not None


def log_processed(out_con, source_db: str, source_id: str, rules_found: int, status: str, error: Optional[str] = None):
    out_con.execute(
        """
        INSERT OR REPLACE INTO process_log_v2 VALUES (?, ?, ?, ?, ?, ?)
        """,
        [source_db, source_id, rules_found, status, error, now_utc()],
    )


def save_source_rules(
    out_con,
    source_db: str,
    source_id: str,
    source_url: Optional[str],
    repo_name: Optional[str],
    rule_source: Optional[str],
    rules: list[dict],
    model: str,
):
    out_con.execute(
        "DELETE FROM processed_rules_v2 WHERE source_db = ? AND source_id = ?",
        [source_db, source_id],
    )

    inserted = 0
    for i, rule in enumerate(rules):
        rule = normalize_prediction(rule, include_rule_text=True)
        rule_text = rule.get("rule_text", "")
        if not rule_text:
            continue
        rid = hashlib.sha256(
            f"{source_db}:{source_id}:{i}:{rule_text}".encode()
        ).hexdigest()[:16]
        out_con.execute(
            """
            INSERT OR REPLACE INTO processed_rules_v2 VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                rid,
                source_db,
                source_id,
                source_url,
                repo_name,
                rule_source,
                rule_text,
                i,
                rule.get("rule_type"),
                rule.get("enforcement_mode"),
                rule.get("mechanism"),
                json.dumps(rule.get("labels", [])),
                rule.get("confidence"),
                rule.get("reasoning"),
                False,
                model,
                model,
                now_utc(),
            ],
        )
        inserted += 1
    return inserted


def extract_rules(client: openai.OpenAI, content: str, model: str, prompt: str) -> list[dict]:
    truncated = content[:MAX_CONTENT_CHARS]
    if len(content) > MAX_CONTENT_CHARS:
        truncated += "\n\n[...truncated...]"

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": truncated},
        ],
    )

    raw = strip_code_fences(response.choices[0].message.content)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array from extraction response")
    return data


def classify_rule(client: openai.OpenAI, rule_text: str, model: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": rule_text},
        ],
    )
    raw = strip_code_fences(response.choices[0].message.content)
    data = json.loads(raw)
    return normalize_prediction(data, include_rule_text=False)


def process_source(
    db_key: str,
    db_path: str,
    out_con: duckdb.DuckDBPyConnection,
    client: openai.OpenAI,
    model: str,
    prompt: str,
    limit: int,
    reprocess: bool,
    workers: int,
) -> dict:
    if not os.path.exists(db_path):
        print(f"  [{db_key}] not found, skipping")
        return {"ok": 0, "err": 0, "skipped": 0, "empty": 0}

    try:
        src_con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:
        print(f"  [{db_key}] cannot open: {exc}")
        return {"ok": 0, "err": 0, "skipped": 0, "empty": 0}

    query = "SELECT id, source_url, repo_name, content, source FROM rules ORDER BY content_len DESC NULLS LAST"
    if limit:
        query += f" LIMIT {limit}"

    try:
        rows = src_con.execute(query).fetchall()
    except Exception as exc:
        print(f"  [{db_key}] query failed: {exc}")
        src_con.close()
        return {"ok": 0, "err": 0, "skipped": 0, "empty": 0}

    src_con.close()

    todo = []
    skipped = 0
    empty = 0

    for source_id, source_url, repo_name, content, rule_source in rows:
        if not reprocess and already_processed(out_con, db_key, source_id):
            skipped += 1
            continue
        if not content or len(content.strip()) < MIN_CONTENT_CHARS:
            log_processed(out_con, db_key, source_id, 0, "empty")
            empty += 1
            continue
        todo.append((source_id, source_url, repo_name, content, rule_source))

    total = len(todo)
    print(f"\n  [{db_key}] {len(rows)} rows - {total} to process, {skipped} skipped, {empty} empty")

    ok = 0
    err = 0
    rate_limit_event = threading.Event()

    def call_llm(row):
        source_id, source_url, repo_name, content, rule_source = row
        while rate_limit_event.is_set():
            time.sleep(1)
        try:
            rules = extract_rules(client, content, model, prompt)
            return ("ok", source_id, source_url, repo_name, rule_source, rules, None)
        except json.JSONDecodeError as exc:
            return ("err", source_id, source_url, repo_name, rule_source, [], f"JSON: {exc}")
        except openai.RateLimitError:
            print(f"  [{db_key}] rate limit - pausing all threads for 60s")
            rate_limit_event.set()
            time.sleep(60)
            rate_limit_event.clear()
            return ("retry", source_id, source_url, repo_name, rule_source, [], None)
        except Exception as exc:
            return ("err", source_id, source_url, repo_name, rule_source, [], str(exc)[:200])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_llm, row): row for row in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            status, source_id, source_url, repo_name, rule_source, rules, error = fut.result()
            label = (source_url or source_id)[:60]
            if status == "ok":
                inserted = save_source_rules(
                    out_con,
                    db_key,
                    source_id,
                    source_url,
                    repo_name,
                    rule_source,
                    rules,
                    model,
                )
                log_processed(out_con, db_key, source_id, inserted, "ok")
                print(f"  [{db_key}] {i}/{total} {label} -> {inserted} rules")
                ok += 1
            elif status == "err":
                log_processed(out_con, db_key, source_id, 0, "error", error)
                print(f"  [{db_key}] {i}/{total} {label} -> ERROR: {error}")
                err += 1
            # retry rows are simply not logged so they can be retried later

    return {"ok": ok, "err": err, "skipped": skipped, "empty": empty}


def selected_rules_query(
    confidence_below: Optional[float],
    selected_ids: list[str],
    include_marked: bool,
    limit: int,
):
    wheres = []
    params = []

    if confidence_below is not None:
        wheres.append("(confidence IS NULL OR confidence < ?)")
        params.append(confidence_below)

    if selected_ids:
        placeholders = ", ".join(["?"] * len(selected_ids))
        wheres.append(f"id IN ({placeholders})")
        params.extend(selected_ids)

    if include_marked:
        wheres.append("manual_reprocess = TRUE")

    if not wheres:
        raise ValueError("no rules selected; add --confidence-below, --id/--ids-file, or marked rows")

    where_sql = " OR ".join(wheres)
    query = f"""
        SELECT id, rule_text
        FROM processed_rules_v2
        WHERE {where_sql}
        ORDER BY COALESCE(confidence, 0.0) ASC, id
    """
    if limit:
        query += f" LIMIT {limit}"
    return query, params


def update_rule_classification(
    out_con: duckdb.DuckDBPyConnection,
    rule_id: str,
    prediction: dict,
    model: str,
):
    out_con.execute(
        """
        UPDATE processed_rules_v2
        SET rule_type = ?,
            enforcement_mode = ?,
            mechanism = ?,
            labels_json = ?,
            confidence = ?,
            reasoning = ?,
            manual_reprocess = FALSE,
            classification_model = ?,
            updated_at = ?
        WHERE id = ?
        """,
        [
            prediction.get("rule_type"),
            prediction.get("enforcement_mode"),
            prediction.get("mechanism"),
            json.dumps(prediction.get("labels", [])),
            prediction.get("confidence"),
            prediction.get("reasoning"),
            model,
            now_utc(),
            rule_id,
        ],
    )


def reclassify_rules(
    out_con: duckdb.DuckDBPyConnection,
    client: openai.OpenAI,
    model: str,
    prompt: str,
    confidence_below: Optional[float],
    selected_ids: list[str],
    include_marked: bool,
    limit: int,
    workers: int,
) -> dict:
    query, params = selected_rules_query(confidence_below, selected_ids, include_marked, limit)
    rows = out_con.execute(query, params).fetchall()
    total = len(rows)
    if total == 0:
        print("No matching rules selected for reclassification.")
        return {"ok": 0, "err": 0}

    print(f"[reclassify] Selected {total} rules")
    ok = 0
    err = 0
    rate_limit_event = threading.Event()

    def call_llm(row):
        rule_id, rule_text = row
        while rate_limit_event.is_set():
            time.sleep(1)
        try:
            prediction = classify_rule(client, rule_text, model, prompt)
            return ("ok", rule_id, prediction, None)
        except json.JSONDecodeError as exc:
            return ("err", rule_id, None, f"JSON: {exc}")
        except openai.RateLimitError:
            print("  [reclassify] rate limit - pausing all threads for 60s")
            rate_limit_event.set()
            time.sleep(60)
            rate_limit_event.clear()
            return ("retry", rule_id, None, None)
        except Exception as exc:
            return ("err", rule_id, None, str(exc)[:200])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_llm, row): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            status, rule_id, prediction, error = fut.result()
            if status == "ok":
                update_rule_classification(out_con, rule_id, prediction, model)
                print(f"  [reclassify] {i}/{total} {rule_id} -> ok")
                ok += 1
            elif status == "err":
                print(f"  [reclassify] {i}/{total} {rule_id} -> ERROR: {error}")
                err += 1
            # retry rows are simply skipped so they can be rerun later

    return {"ok": ok, "err": err}


def mark_rules(out_con: duckdb.DuckDBPyConnection, rule_ids: list[str], clear: bool):
    if not rule_ids:
        raise ValueError("no rule IDs supplied")
    placeholders = ", ".join(["?"] * len(rule_ids))
    value = False if clear else True
    out_con.execute(
        f"UPDATE processed_rules_v2 SET manual_reprocess = ? WHERE id IN ({placeholders})",
        [value] + rule_ids,
    )
    action = "Cleared" if clear else "Marked"
    print(f"{action} {len(rule_ids)} rule(s).")


def print_stats(out_con: duckdb.DuckDBPyConnection, confidence_below: Optional[float]):
    total = out_con.execute("SELECT COUNT(*) FROM processed_rules_v2").fetchone()[0]
    marked = out_con.execute(
        "SELECT COUNT(*) FROM processed_rules_v2 WHERE manual_reprocess = TRUE"
    ).fetchone()[0]

    print(f"Total rules: {total}")
    print(f"Marked for reclassification: {marked}")

    if confidence_below is not None:
        low = out_con.execute(
            "SELECT COUNT(*) FROM processed_rules_v2 WHERE confidence IS NULL OR confidence < ?",
            [confidence_below],
        ).fetchone()[0]
        print(f"Rules below confidence {confidence_below:.2f}: {low}")

    print("\nBy rule type:")
    for row in out_con.execute(
        """
        SELECT COALESCE(rule_type, '(null)'), COUNT(*)
        FROM processed_rules_v2
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall():
        print(f"  {row[0]:20s} {row[1]:>6}")

    print("\nBy enforcement mode:")
    for row in out_con.execute(
        """
        SELECT COALESCE(enforcement_mode, '(null)'), COUNT(*)
        FROM processed_rules_v2
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall():
        print(f"  {row[0]:20s} {row[1]:>6}")

    print("\nTop labels:")
    try:
        rows = out_con.execute(
            """
            SELECT label, COUNT(*) AS n
            FROM (
                SELECT UNNEST(json_extract_string(labels_json, '$[*]')) AS label
                FROM processed_rules_v2
                WHERE labels_json IS NOT NULL
            )
            GROUP BY 1 ORDER BY 2 DESC LIMIT 20
            """
        ).fetchall()
        for row in rows:
            print(f"  {row[0]:25s} {row[1]:>6}")
    except Exception as exc:
        print(f"  (label stats unavailable: {exc})")


def ensure_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return api_key


def build_common_prompt_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--instructions-file",
        help="Plain-text file with extra prompt instructions for this run",
    )
    parser.add_argument(
        "--examples-file",
        help="JSON array of labeled examples to include in the prompt",
    )


def load_prompt_inputs(args) -> tuple[str, list[dict]]:
    extra = load_text_file(args.instructions_file)
    examples = load_examples(args.examples_file)
    return extra, examples


def main():
    parser = argparse.ArgumentParser(
        description="V2 rules processor with initial extraction and iterative in-place reclassification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initial = subparsers.add_parser("initial", help="Initial raw-file extraction and classification")
    initial.add_argument("--out", default=DEFAULT_OUT_DB, help=f"Output DuckDB file (default: {DEFAULT_OUT_DB})")
    initial.add_argument("--source", metavar="DB_PATH", help="Process only this source DB file")
    initial.add_argument("--limit", type=int, default=0, help="Max raw rows to process per source DB (0 = all)")
    initial.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model ID")
    initial.add_argument("--reprocess", action="store_true", help="Reprocess raw source rows already logged in process_log_v2")
    initial.add_argument("--workers", type=int, default=10, help="Parallel LLM calls (default: 10)")
    build_common_prompt_args(initial)

    reclassify = subparsers.add_parser("reclassify", help="Reclassify selected existing atomic rules")
    reclassify.add_argument("--out", default=DEFAULT_OUT_DB, help=f"Output DuckDB file (default: {DEFAULT_OUT_DB})")
    reclassify.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model ID")
    reclassify.add_argument("--confidence-below", type=float, help="Select rules with confidence below this threshold")
    reclassify.add_argument("--id", action="append", default=[], help="Specific rule ID to reclassify; may be repeated")
    reclassify.add_argument("--ids-file", help="File containing rule IDs, one per line or comma-separated")
    reclassify.add_argument("--skip-marked", action="store_true", help="Do not include rows with manual_reprocess = true")
    reclassify.add_argument("--limit", type=int, default=0, help="Max number of selected rules to reclassify (0 = all)")
    reclassify.add_argument("--workers", type=int, default=10, help="Parallel LLM calls (default: 10)")
    build_common_prompt_args(reclassify)

    mark = subparsers.add_parser("mark", help="Mark or unmark specific rules for later reclassification")
    mark.add_argument("--out", default=DEFAULT_OUT_DB, help=f"Output DuckDB file (default: {DEFAULT_OUT_DB})")
    mark.add_argument("--id", action="append", default=[], help="Specific rule ID to mark; may be repeated")
    mark.add_argument("--ids-file", help="File containing rule IDs, one per line or comma-separated")
    mark.add_argument("--clear", action="store_true", help="Clear the mark instead of setting it")

    stats = subparsers.add_parser("stats", help="Show quick counts for processed_v2 data")
    stats.add_argument("--out", default=DEFAULT_OUT_DB, help=f"Output DuckDB file (default: {DEFAULT_OUT_DB})")
    stats.add_argument("--confidence-below", type=float, help="Also count rules below this confidence threshold")

    args = parser.parse_args()

    con = duckdb.connect(getattr(args, "out", DEFAULT_OUT_DB))
    init_out_db(con)

    if args.command == "initial":
        api_key = ensure_api_key()
        extra, examples = load_prompt_inputs(args)
        prompt = build_extract_prompt(extra, examples)
        client = openai.OpenAI(api_key=api_key)

        if args.source:
            key = os.path.splitext(os.path.basename(args.source))[0]
            sources = {key: os.path.abspath(args.source)}
        else:
            sources = SOURCE_DBS

        totals = {"ok": 0, "err": 0, "skipped": 0, "empty": 0}
        print(f"[initial] Output: {args.out}  Model: {args.model}")

        for db_key, db_path in sources.items():
            counts = process_source(
                db_key,
                db_path,
                con,
                client,
                args.model,
                prompt,
                args.limit,
                args.reprocess,
                args.workers,
            )
            for key in totals:
                totals[key] += counts[key]

        total_rules = con.execute("SELECT COUNT(*) FROM processed_rules_v2").fetchone()[0]
        print(f"\n{'=' * 52}")
        print(
            f"Processed: {totals['ok']} ok  {totals['err']} errors  "
            f"{totals['skipped']} skipped  {totals['empty']} empty"
        )
        print(f"Total individual rules in {args.out}: {total_rules}")
        print_stats(con, None)

    elif args.command == "reclassify":
        api_key = ensure_api_key()
        extra, examples = load_prompt_inputs(args)
        prompt = build_reclassify_prompt(extra, examples)
        client = openai.OpenAI(api_key=api_key)
        selected_ids = parse_ids(args.id, args.ids_file)

        result = reclassify_rules(
            con,
            client,
            args.model,
            prompt,
            args.confidence_below,
            selected_ids,
            include_marked=not args.skip_marked,
            limit=args.limit,
            workers=args.workers,
        )
        print(f"\nReclassified: {result['ok']} ok  {result['err']} errors")
        print_stats(con, args.confidence_below)

    elif args.command == "mark":
        rule_ids = parse_ids(args.id, args.ids_file)
        mark_rules(con, rule_ids, clear=args.clear)

    elif args.command == "stats":
        print_stats(con, args.confidence_below)

    con.close()


if __name__ == "__main__":
    main()
