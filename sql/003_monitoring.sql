-- Monitoring fields added after the initial schema was deployed.
ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS cached_tokens INT;
