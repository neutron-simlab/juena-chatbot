-- Apply once before starting the recovery-capable application on an existing DB.
-- Stop the old application first: its shutdown still fails active jobs globally.
BEGIN;
ALTER TABLE research_jobs
    ADD COLUMN IF NOT EXISTS run_input JSON NOT NULL DEFAULT '{}';
ALTER TABLE research_jobs ALTER COLUMN run_input DROP DEFAULT;

-- Old active jobs never saved their inputs or execution checkpoints.
UPDATE research_jobs
SET status = 'failed', finished_at = now(),
    error = 'This job predates research recovery and must be started again'
WHERE status IN ('queued', 'running') AND run_input->>'provider' IS NULL;
COMMIT;
