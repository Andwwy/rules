# Rule Annotator

Interactive web app for annotating rules in agent/LLM config files: select text to
add rules, tag them (PROHIBITION / PRESCRIPTION / PERMISSION / PREFERENCE), run an
editable **LLM judge** prompt to get a rationale, leave comments, switch between
annotators, filter the rule list and document highlights by label source
(**All / Human / LLM**), and export per-annotator to CSV.

## Database — MotherDuck (cloud) or local

Both the source documents (`sample`) and the annotations live in **MotherDuck**
(cloud DuckDB) whenever `MOTHERDUCK_TOKEN` is set — and it should be set everywhere,
**including local dev**, so everyone reads and writes the same shared cloud DB. This
is what makes the Render deployment and multi-device use work. If the token is
absent the app falls back to local files (`sample.db` / `annotations.db`) — that's
an offline-only escape hatch, not the normal path.

| Env var | Purpose |
|---|---|
| `MOTHERDUCK_TOKEN` | MotherDuck access token. **Set this everywhere (local + Render)** to use the shared cloud DB. Get it from the MotherDuck dashboard. |
| `MOTHERDUCK_DATABASE` | MotherDuck database name (default `rules`). Created automatically if missing. |
| `PERPLEXITY_API_KEY` | Powers the LLM judge — Sonar models via the Chat API, `anthropic/*` / `openai/*` models via Perplexity's Agent API. |

- **Locally:** put `MOTHERDUCK_TOKEN=...` in `.env` (same file as `PERPLEXITY_API_KEY`) — then local dev reads/writes MotherDuck just like prod. Only omit it if you deliberately want offline local files.
- **Render:** add `MOTHERDUCK_TOKEN` (and optionally `MOTHERDUCK_DATABASE`) in the service's Environment settings. `PERPLEXITY_API_KEY` goes there too. (`render.yaml` declares these with `sync:false`, so Render prompts for the values on first deploy.)

**Move existing local annotations into MotherDuck** (one-off, idempotent):

```bash
python3 migrate_to_motherduck.py    # reads MOTHERDUCK_TOKEN from .env / env
```

## Deploy to Render

`render.yaml` is a [Render Blueprint](https://render.com/docs/blueprint-spec) that
runs the Flask app as a web service on Render's native Python runtime: it installs
`requirements.txt` and serves `annotate:app` with gunicorn on Render's `$PORT`.

1. Push this repo to GitHub/GitLab and create a **Blueprint** in the Render
   dashboard pointing at it (or **New → Web Service** and let it detect `render.yaml`).
2. When prompted, set `MOTHERDUCK_TOKEN` and `PERPLEXITY_API_KEY` (declared
   `sync:false` so they're never committed). `MOTHERDUCK_DATABASE` defaults to `rules`.

No database ships in git. Both the **source documents** (the `sample` table) and the
**annotations** are served from MotherDuck, so Render only needs the code plus the
`MOTHERDUCK_TOKEN`. DuckDB's extension dir is pointed at the writable `/tmp` when the
`RENDER` env var is present (set automatically by Render).

The annotation set is currently a **fixed 100-file sample** loaded into MotherDuck
(`rules.sample`). Load/refresh it from the local `sample.db` with:

```bash
python3 upload_sample_to_motherduck.py   # reads MOTHERDUCK_TOKEN from .env / env
```

> **Note:** this 100-file sample is a stopgap — we'll switch to the full set of
> scraped files later. See [SPEC.md](SPEC.md).

### Concurrency model (why `--workers 1 --threads 8`)

The app holds **one** cached MotherDuck session per process (lock-guarded lazy
init) and gives each request its own cursor — cursors are DuckDB's thread-safe
unit. Threads share that session cheaply, so scale with **threads**, not workers:
extra gunicorn workers would each open their own MotherDuck session. Keep
`--workers 1` if you tune the start command in `render.yaml`.

## Run with Docker (recommended — keeps running)

```bash
docker compose up -d --build
```

Then open http://localhost:5002.

- `.env` (with `MOTHERDUCK_TOKEN` and `PERPLEXITY_API_KEY`) is read from the project
  directory via the bind mount, so the container reads/writes MotherDuck like
  everywhere else. (Without the token it falls back to the bind-mounted local
  `sample.db` / `annotations.db`.)
- The container has `restart: unless-stopped`, so it comes back after crashes or a
  Docker/host restart.

> **Editing the code?** The Flask reloader is **off** by default, so the running
> container keeps serving the code it loaded at startup — a browser refresh is not
> enough. Either `docker compose restart` after each change (the bind mount means
> no rebuild is needed), or set `DEBUG=1` in `docker-compose.yml` to enable
> auto-reload during development.

Common commands:

```bash
docker compose logs -f        # follow logs
docker compose restart        # restart (also picks up code edits)
docker compose down           # stop and remove the container
```

CSV export is at the **⬇ CSV** button (top-right of the inspector) or directly at
http://localhost:5002/export. It is one flat file covering **both entity types** —
filter on the leading `record_type` column (`rule` / `context` / `relation`):

- **rule / context rows** fill `kind`, `tag`, `rule_text`, `line_start`/`line_end`,
  `char_start`/`char_end`, `user_comment`, and `llm_rationale` (human annotation and
  LLM judge output side-by-side), plus the rule `id`.
- **relation rows** fill `relation_type` (the annotator's label),
  `llm_relation_type` (the LLM's suggested label, for contrast),
  `source_rule_text`/`target_rule_text` (the endpoints resolved to text), and
  `source_id`/`target_id`.

Every row also carries `annotator`, `source_url`, `source` (`hand` / `llm` /
`revise`), and `created_at`. Scope is the current file or all labeled files, per
the chooser under the button.

## Run directly (without Docker)

```bash
pip install -r requirements.txt
python annotate.py --port 5002        # add --debug for the Flask reloader
```

`HOST`, `PORT`, and `DEBUG` can also be set as environment variables.

> **DuckDB version:** `requirements.txt` pins `duckdb==1.5.2` to match the version
> that wrote the existing `.db` files. DuckDB won't open a database created by a
> newer minor version, so keep this in sync if you regenerate them.
