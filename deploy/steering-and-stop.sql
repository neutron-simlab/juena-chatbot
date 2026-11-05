-- Additive migration. Drain foreground runs and stop API/worker instances first.
-- Install the matching API and sandbox worker revision before restarting them.
BEGIN;
ALTER TABLE IF EXISTS sandbox_jobs ADD COLUMN IF NOT EXISTS run_id varchar(255);
ALTER TABLE IF EXISTS sandbox_jobs ADD COLUMN IF NOT EXISTS cancel_requested boolean NOT NULL DEFAULT false;
DO $$ BEGIN
    IF to_regclass('sandbox_jobs') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS ix_sandbox_jobs_run_id ON sandbox_jobs (run_id);
    END IF;
END $$;
COMMIT;
