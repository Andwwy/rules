# Rules

This repo now centers on the V2 rule-classification flow.

The files that matter most are:

- `process_v2.py`: extracts and classifies rules into the V2 taxonomy
- `processed_v2.db`: current V2 output database
- `index.html`: read-only browser for `processed_v2.db`
- `serve.py`: tiny static server for the browser
- `app_v2.py`: optional Flask UI for marking rules and rerunning classification

## Setup

```bash
cd /Users/jennyma/Projects/rules
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI key in `.env`:

```env
OPENAI_API_KEY=your_key_here
```

## Run V2 Classification

Small test run:

```bash
python3 process_v2.py initial --limit 100 --workers 5
```

Full run:

```bash
python3 process_v2.py initial
```

Reclassify low-confidence rules:

```bash
python3 process_v2.py reclassify --confidence-below 0.7
```

Mark specific rules for rerun:

```bash
python3 process_v2.py mark --id RULE_ID_HERE
python3 process_v2.py reclassify
```

## View The Results

Read-only browser:

```bash
python3 serve.py
```

Then open:

```text
http://localhost:5001
```

Optional mark-and-rerun UI:

```bash
python3 app_v2.py
```

Then open:

```text
http://127.0.0.1:5002
```

## V2 Taxonomy

- `rule_type`
  - `output_rule`
  - `process_rule`

- `enforcement_mode`
  - `deterministic`
  - `llm`
  - `mixed`

`output_rule` means the rule constrains the produced artifact: response, code, files, names, imports, docs, or repo structure.

`process_rule` means the rule constrains behavior over time: planning, approvals, retries, tool use, or action ordering.

`deterministic` means the rule can be checked from available runtime/tool/file signals.

`llm` means semantic judgment is still needed even with those signals.

`mixed` means part is checkable mechanically and part still needs judgment.

## Notes

- `process_v2.py` already contains built-in examples for the current taxonomy.
- The static browser only shows `processed_v2.db`.
- The old V1 browser was removed.
