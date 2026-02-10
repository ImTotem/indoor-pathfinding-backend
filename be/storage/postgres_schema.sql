-- PostgreSQL Schema for SLAM Service
-- Purpose: Store SLAM job queue state and binary database files
-- Created: 2026-02-10
-- Note: Pure PostgreSQL SQL - no ORM or migration framework dependencies

-- ============================================================================
-- TABLE: slam_jobs
-- Purpose: Queue tracking for SLAM processing jobs
-- Tracks job lifecycle: pending → in_progress → success/failed
-- ============================================================================
CREATE TABLE slam_jobs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    map_id VARCHAR(255) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    
    -- Constraints
    CONSTRAINT status_check CHECK (status IN ('pending', 'in_progress', 'success', 'failed'))
);

-- Indexes for query performance
CREATE INDEX idx_slam_jobs_session_id ON slam_jobs(session_id);
CREATE INDEX idx_slam_jobs_map_id ON slam_jobs(map_id);
CREATE INDEX idx_slam_jobs_status ON slam_jobs(status);

-- Column documentation
COMMENT ON TABLE slam_jobs IS 'Queue tracking for SLAM processing jobs. Stores job state and lifecycle information.';
COMMENT ON COLUMN slam_jobs.id IS 'Auto-incrementing primary key';
COMMENT ON COLUMN slam_jobs.session_id IS 'Session identifier from FastAPI routes (string)';
COMMENT ON COLUMN slam_jobs.map_id IS 'Unique map identifier (UUID string, generated server-side)';
COMMENT ON COLUMN slam_jobs.status IS 'Job status: pending, in_progress, success, or failed';
COMMENT ON COLUMN slam_jobs.created_at IS 'Timestamp when job was created';
COMMENT ON COLUMN slam_jobs.updated_at IS 'Timestamp when job was last updated';
COMMENT ON COLUMN slam_jobs.error_message IS 'Error details if job failed (NULL if successful)';

-- ============================================================================
-- TABLE: slam_databases
-- Purpose: Binary storage for SLAM database files (.db files)
-- Stores both input.db (raw SLAM data) and output.db (processed results)
-- ============================================================================
CREATE TABLE slam_databases (
    id SERIAL PRIMARY KEY,
    map_id VARCHAR(255) NOT NULL,
    db_type VARCHAR(20) NOT NULL,
    db_data BYTEA NOT NULL,
    size_bytes BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT db_type_check CHECK (db_type IN ('input', 'output')),
    CONSTRAINT unique_map_db_type UNIQUE (map_id, db_type)
);

-- Indexes for query performance
CREATE INDEX idx_slam_databases_map_id ON slam_databases(map_id);
CREATE INDEX idx_slam_databases_db_type ON slam_databases(db_type);

-- Column documentation
COMMENT ON TABLE slam_databases IS 'Binary storage for SLAM database files. Stores input.db and output.db as BYTEA blobs. Supports files up to 73MB.';
COMMENT ON COLUMN slam_databases.id IS 'Auto-incrementing primary key';
COMMENT ON COLUMN slam_databases.map_id IS 'Foreign key reference to slam_jobs.map_id';
COMMENT ON COLUMN slam_databases.db_type IS 'Database type: input (raw SLAM data) or output (processed results)';
COMMENT ON COLUMN slam_databases.db_data IS 'Binary database file content (BYTEA format, max ~73MB)';
COMMENT ON COLUMN slam_databases.size_bytes IS 'Size of db_data in bytes (for monitoring and quota management)';
COMMENT ON COLUMN slam_databases.created_at IS 'Timestamp when database file was stored';

-- ============================================================================
-- EXAMPLE QUERIES (for reference)
-- ============================================================================

-- Example 1: Insert a new SLAM job
-- INSERT INTO slam_jobs (session_id, map_id, status)
-- VALUES ('session_abc123', 'map_uuid_001', 'pending');

-- Example 2: Update job status to in_progress
-- UPDATE slam_jobs
-- SET status = 'in_progress', updated_at = CURRENT_TIMESTAMP
-- WHERE map_id = 'map_uuid_001';

-- Example 3: Mark job as failed with error message
-- UPDATE slam_jobs
-- SET status = 'failed', error_message = 'ORB-SLAM3 initialization failed', updated_at = CURRENT_TIMESTAMP
-- WHERE map_id = 'map_uuid_001';

-- Example 4: Query pending jobs (for job queue worker)
-- SELECT id, session_id, map_id FROM slam_jobs
-- WHERE status = 'pending'
-- ORDER BY created_at ASC
-- LIMIT 1;

-- Example 5: Store input database file
-- INSERT INTO slam_databases (map_id, db_type, db_data, size_bytes)
-- VALUES ('map_uuid_001', 'input', E'\\x...binary_data...', 5242880);

-- Example 6: Retrieve output database file
-- SELECT db_data, size_bytes FROM slam_databases
-- WHERE map_id = 'map_uuid_001' AND db_type = 'output';

-- Example 7: Query all jobs for a session
-- SELECT id, map_id, status, created_at, updated_at
-- FROM slam_jobs
-- WHERE session_id = 'session_abc123'
-- ORDER BY created_at DESC;

-- Example 8: Check database storage usage
-- SELECT
--     COUNT(*) as total_files,
--     SUM(size_bytes) as total_bytes,
--     ROUND(SUM(size_bytes)::numeric / 1024 / 1024 / 1024, 2) as total_gb
-- FROM slam_databases;
