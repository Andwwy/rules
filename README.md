# Rule Annotator

Interactive web app for annotating rules in agent/LLM config files: select text to
add rules, tag them (PROHIBITION / PRESCRIPTION / PERMISSION / PREFERENCE), run an
editable **LLM judge** prompt to get a rationale, leave comments, switch between
annotators, and export per-annotator to CSV.

## Database — MotherDuck (cloud) or local

Annotations are stored in **MotherDuck** (cloud DuckDB) when a token is present,
which is what makes the Vercel deployment and multi-device use work. With no token
it falls back to a local `annotations.db` file.

| Env var | Purpose |
|---|---|
| `MOTHERDUCK_TOKEN` | MotherDuck access token. **Set this to use the cloud DB.** Get it from the MotherDuck dashboard. |
| `MOTHERDUCK_DATABASE` | MotherDuck database name (default `rules`). Created automatically if missing. |

- **Locally:** put `MOTHERDUCK_TOKEN=...` in `.env` (same file as `PERPLEXITY_API_KEY`). Omit it to use the local `annotations.db` instead.
- **Vercel:** add `MOTHERDUCK_TOKEN` (and optionally `MOTHERDUCK_DATABASE`) in Project → Settings → Environment Variables. `PERPLEXITY_API_KEY` goes there too.

**Move existing local annotations into MotherDuck** (one-off, idempotent):

```bash
python3 migrate_to_motherduck.py    # reads MOTHERDUCK_TOKEN from .env / env
```

## Deploy to Vercel

`vercel.json` + `api/index.py` are set up to run the Flask app on Vercel's Python
runtime (all routes rewrite to the function; `annotate.py` and `sample.db` are
bundled via `includeFiles`).

```bash
vercel            # preview deploy
vercel --prod     # production
```

Set the env vars (`MOTHERDUCK_TOKEN`, `PERPLEXITY_API_KEY`) in the Vercel dashboard
first. Vercel's read-only filesystem is handled — DuckDB's extension dir is pointed
at `/tmp` when the `VERCEL` env var is present.

> **Note:** `sample.db` (the source documents) is bundled into the deployment. The
> large `rules.db` / `processed.db` files are excluded via `.vercelignore`.

## Run with Docker (recommended — keeps running, persists annotations)

```bash
docker compose up -d --build
```

Then open http://localhost:5002.

- Annotations are written to `annotations.db` on the host (bind-mounted), so they
  survive container restarts.
- `sample.db` (the source files) and `.env` (for `PERPLEXITY_API_KEY`) are read from
  the project directory via the same bind mount.
- The container has `restart: unless-stopped`, so it comes back after crashes or a
  Docker/host restart.

Common commands:

```bash
docker compose logs -f        # follow logs
docker compose restart        # restart (annotations persist)
docker compose down           # stop and remove the container
```

CSV export is at the **⬇ CSV** button (top-right of the inspector) or directly at
http://localhost:5002/export. It includes the `tag`, `user_comment`, and
`llm_rationale` columns so the human annotation and the LLM judge output sit
side-by-side.

## Run directly (without Docker)

```bash
pip install -r requirements.txt
python annotate.py --port 5002        # add --debug for the Flask reloader
```

`HOST`, `PORT`, and `DEBUG` can also be set as environment variables.

> **DuckDB version:** `requirements.txt` pins `duckdb==1.5.2` to match the version
> that wrote the existing `.db` files. DuckDB won't open a database created by a
> newer minor version, so keep this in sync if you regenerate them.
