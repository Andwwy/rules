"""Non-destructive migration to the shared-batch + per-user-label model.

Run INSIDE the app container (it has MOTHERDUCK_TOKEN):
    docker exec -i rule-annotator python migrate_batches.py

For each (file_id, annotator) that still has un-migrated rows it:
  1. creates one shared `batches` row (creator = that annotator; mode 'revise' if any revise
     rows else 'extract'),
  2. stamps `batch_id` on that annotator's rows (KEEPS ids stable — relations still resolve),
  3. sets `owner` on the annotator's hand rows,
  4. copies the per-user label state (tag/base_tag/llm_tag/reviewed/context_type/notes) into the
     new per-user `rule_labels` table,
  5. stamps `batch_id` on that annotator's relations.

Idempotent: only touches rows WHERE batch_id IS NULL, so re-running is a no-op. The original
extracted_rules columns are left intact (the snapshot is the undo).
"""
import annotate as A

assert A.MOTHERDUCK_DATABASE, "MOTHERDUCK_DATABASE must be set (expected rules_v2_6_18)"
print("DB:", A.MOTHERDUCK_DATABASE)

con = A.annot_con()
pairs = con.execute(
    "SELECT DISTINCT file_id, annotator FROM extracted_rules WHERE batch_id IS NULL"
).fetchall()
print("(file, annotator) pairs to migrate:", len(pairs))

for fid, ann in pairs:
    has_rev = con.execute(
        "SELECT count(*) FROM extracted_rules WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?"
        " AND batch_id IS NULL AND source='revise'", [fid, ann]).fetchone()[0]
    mode = "revise" if has_rev else "extract"
    bid = A.make_id(fid, ann or "", "migration")   # deterministic per (file, annotator) → idempotent
    con.execute("INSERT OR IGNORE INTO batches(batch_id,file_id,mode,creator,created_at)"
                " VALUES (?,?,?,?,now())", [bid, fid, mode, ann])
    con.execute("UPDATE extracted_rules SET batch_id=? WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?"
                " AND batch_id IS NULL", [bid, fid, ann])
    con.execute("UPDATE extracted_rules SET owner=? WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?"
                " AND batch_id=? AND source='hand' AND owner IS NULL", [ann, fid, ann, bid])
    con.execute(
        "INSERT OR IGNORE INTO rule_labels(batch_id,rule_id,annotator,tag,base_tag,llm_tag,reviewed,context_type,notes,updated_at)"
        " SELECT batch_id, id, ?, tag, base_tag, llm_tag, reviewed, context_type, notes, now()"
        " FROM extracted_rules WHERE file_id=? AND annotator IS NOT DISTINCT FROM ? AND batch_id=?"
        "   AND (tag IS NOT NULL OR base_tag IS NOT NULL OR llm_tag IS NOT NULL OR reviewed IS NOT NULL"
        "        OR context_type IS NOT NULL OR notes IS NOT NULL)",
        [ann or "", fid, ann, bid])
    con.execute("UPDATE relations SET batch_id=? WHERE file_id=? AND annotator IS NOT DISTINCT FROM ?"
                " AND batch_id IS NULL", [bid, fid, ann])
    print(f"  migrated {fid} / {ann} -> batch {bid} ({mode})")

# ── backfill batch models from the original judge_runs, and LINK the old runs/votes to the
#    batch (so the run history shows models + the vote rings/candidates resolve) ──
for bid, fid, mode, creator in con.execute(
        "SELECT batch_id, file_id, mode, creator FROM batches WHERE models IS NULL").fetchall():
    jr = con.execute("SELECT id, models, threshold, judge_total FROM judge_runs WHERE file_id=?"
                     " AND annotator IS NOT DISTINCT FROM ? AND mode=? ORDER BY created_at DESC LIMIT 1",
                     [fid, creator, mode]).fetchone()
    if not jr:
        jr = con.execute("SELECT id, models, threshold, judge_total FROM judge_runs WHERE file_id=?"
                         " AND annotator IS NOT DISTINCT FROM ? AND mode IN ('extract','revise')"
                         " ORDER BY created_at DESC LIMIT 1", [fid, creator]).fetchone()
    if jr:
        con.execute("UPDATE batches SET models=?, threshold=?, judge_total=? WHERE batch_id=?",
                    [jr[1], jr[2], jr[3], bid])
        con.execute("UPDATE judge_runs SET batch_id=? WHERE id=?", [bid, jr[0]])
        con.execute("UPDATE judge_votes SET batch_id=? WHERE run_id=?", [bid, jr[0]])

# ── verification ────────────────────────────────────────────────────────────
left = con.execute("SELECT count(*) FROM extracted_rules WHERE batch_id IS NULL").fetchone()[0]
nbatch = con.execute("SELECT count(*) FROM batches").fetchone()[0]
nlab = con.execute("SELECT count(*) FROM rule_labels").fetchone()[0]
relleft = con.execute("SELECT count(*) FROM relations WHERE batch_id IS NULL").fetchone()[0]
# every relation endpoint still resolves to a real rule
orphans = con.execute(
    "SELECT count(*) FROM relations r WHERE NOT EXISTS (SELECT 1 FROM extracted_rules e WHERE e.id=r.source_id)"
    " OR NOT EXISTS (SELECT 1 FROM extracted_rules e WHERE e.id=r.target_id)").fetchone()[0]
print(f"\nVERIFY: rules without batch_id={left} (expect 0) | batches={nbatch} | rule_labels={nlab}"
      f" | relations without batch_id={relleft} (expect 0) | orphan relation endpoints={orphans} (expect 0)")
