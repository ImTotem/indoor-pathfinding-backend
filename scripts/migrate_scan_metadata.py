"""scan_metadata.db (sqlite) → PG migration helper.

Usage (in server container):
  python /app/scripts/migrate_scan_metadata.py <scan_id_lower> <storage_path_dir_name>
"""
import sys
import sqlite3
import psycopg


def main() -> None:
    scan_id = sys.argv[1]
    storage_dir = sys.argv[2]

    src = sqlite3.connect(f"/app/var/storage/scans/{storage_dir}/scan_metadata.db")
    dst = psycopg.connect("host=db port=5432 user=indoor password=indoor dbname=indoor")
    cur = dst.cursor()

    sess = src.execute(
        "SELECT id, started_at, ended_at, device_model, app_version, state, keyframe_count, notes FROM scan_session"
    ).fetchone()
    cur.execute(
        """
        INSERT INTO scan_session (scan_id, started_at, ended_at, device_model, app_version, state, keyframe_count, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scan_id) DO UPDATE SET keyframe_count=EXCLUDED.keyframe_count, notes=EXCLUDED.notes
        """,
        (scan_id, sess[1], sess[2], sess[3], sess[4], sess[5], sess[6], sess[7]),
    )
    print(f"  scan_session: 1")

    cnt = 0
    for r in src.execute(
        "SELECT scan_id, seq, captured_at, image_path, pose_matrix, tx, ty, tz, tracking_state, rtabmap_node_id FROM keyframe_meta"
    ):
        cur.execute(
            """
            INSERT INTO keyframe_meta (scan_id, seq, captured_at, image_path, pose_matrix, tx, ty, tz, tracking_state, rtabmap_node_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (scan_id, seq) DO NOTHING
            """,
            (scan_id, r[1], r[2], r[3] or "", bytes(r[4]), r[5], r[6], r[7], r[8], r[9]),
        )
        cnt += 1
    print(f"  keyframe_meta: {cnt}")

    cnt = 0
    for r in src.execute(
        "SELECT id, keyframe_seq, created_at, pose_matrix, tx, ty, tz, label, source FROM poi_mark"
    ):
        cur.execute(
            """
            INSERT INTO poi_mark (scan_id, keyframe_seq, created_at, pose_matrix, tx, ty, tz, label, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::poi_source)
            ON CONFLICT DO NOTHING
            """,
            (scan_id, r[1], r[2], bytes(r[3]), r[4], r[5], r[6], r[7], r[8]),
        )
        cnt += 1
    print(f"  poi_mark: {cnt}")

    cnt = 0
    for r in src.execute(
        "SELECT id, keyframe_seq, created_at, pose_matrix, tx, ty, tz, node_type FROM branch_mark"
    ):
        cur.execute(
            """
            INSERT INTO branch_mark (scan_id, keyframe_seq, created_at, pose_matrix, tx, ty, tz)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (scan_id, r[1], r[2], bytes(r[3]), r[4], r[5], r[6]),
        )
        cnt += 1
    print(f"  branch_mark: {cnt} (node_type 정보 유실 — PG schema v8 미반영)")

    cnt = 0
    for r in src.execute(
        "SELECT id, keyframe_seq, created_at, connector_type, prefix, pose_matrix, tx, ty, tz FROM interfloor_mark"
    ):
        cur.execute(
            """
            INSERT INTO interfloor_mark (scan_id, keyframe_seq, created_at, connector_type, prefix, pose_matrix, tx, ty, tz)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (scan_id, r[1], r[2], r[3], r[4], bytes(r[5]), r[6], r[7], r[8]),
        )
        cnt += 1
    print(f"  interfloor_mark: {cnt}")

    dst.commit()
    print("commit done")


if __name__ == "__main__":
    main()
