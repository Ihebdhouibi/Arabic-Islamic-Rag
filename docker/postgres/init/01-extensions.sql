-- Enabled on first container start (empty data volume) via /docker-entrypoint-initdb.d.
-- pg_trgm backs trigram/fuzzy matching on metadata text (book/author titles).
CREATE EXTENSION IF NOT EXISTS pg_trgm;
