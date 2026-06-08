# Rule Annotator

Interactive web app for annotating rules in agent/LLM config files: select text to
add rules, tag them (PROHIBIT / OBLIGE / PERMIT / POWER→norm·strategy), run an
editable **LLM judge** prompt to get a rationale, leave comments, and export to CSV.

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
