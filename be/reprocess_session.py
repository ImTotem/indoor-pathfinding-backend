"""Reprocess existing session with updated SLAM parameters."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from slam_engines.rtabmap.engine import RTABMapEngine
from slam_engines.rtabmap.db_builder import build_database
from slam_engines.rtabmap import constants
import json


async def reprocess_session(session_path: str):
    session = Path(session_path)
    
    if not session.exists():
        print(f"Error: Session not found: {session_path}")
        return
    
    print(f"[Reprocess] Session: {session.name}")
    
    metadata_file = session / "metadata.json"
    if not metadata_file.exists():
        print(f"Error: metadata.json not found")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    intrinsics = metadata.get("camera_intrinsics", {})
    if not intrinsics:
        print("[Reprocess] No intrinsics in metadata, extracting from chunk...")
        chunks_dir = session / "chunks"
        chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
        if not chunk_files:
            print("Error: No chunk files found")
            return
        
        with open(chunk_files[0]) as f:
            chunk = json.load(f)
            if chunk and len(chunk) > 0:
                intrinsics = chunk[0].get("camera_intrinsics", {})
        
        if not intrinsics:
            print("Error: No camera intrinsics found")
            return
    
    print(f"[Reprocess] Camera: {intrinsics['width']}x{intrinsics['height']}, fx={intrinsics['fx']:.1f}")
    
    input_db_old = session / "rtabmap_input.db"
    if input_db_old.exists():
        backup = session / "rtabmap_input.db.old"
        print(f"[Reprocess] Backing up old input DB → {backup.name}")
        input_db_old.rename(backup)
    
    output_db_old = session / "rtabmap.db"
    if output_db_old.exists():
        backup = session / "rtabmap.db.old"
        print(f"[Reprocess] Backing up old output DB → {backup.name}")
        output_db_old.rename(backup)
    
    print(f"[Reprocess] Building new input DB with updated parameters...")
    print(f"  - MODE: RGB-D (depth enabled)")
    print(f"  - OdomF2M/ValidDepthRatio: {constants.DEFAULT_PARAMS.get('OdomF2M/ValidDepthRatio', 'N/A')}")
    print(f"  - Vis/MinInliers: {constants.DEFAULT_PARAMS['Vis/MinInliers']}")
    print(f"  - RGBD/LinearUpdate: {constants.DEFAULT_PARAMS['RGBD/LinearUpdate']}")
    print(f"  - RGBD/AngularUpdate: {constants.DEFAULT_PARAMS['RGBD/AngularUpdate']}")
    
    input_db = build_database(
        str(session),
        intrinsics,
        slam_params=constants.DEFAULT_PARAMS,
        monocular=False
    )
    
    print(f"[Reprocess] Running RTAB-Map reprocess...")
    engine = RTABMapEngine()
    
    output_db = str(session / constants.DATABASE_FILENAME)
    
    input_db_abs = str(Path(input_db).absolute())
    output_db_abs = str(Path(output_db).absolute())
    
    await engine._run_reprocess(input_db_abs, output_db_abs, progress_callback=None)
    
    print(f"[Reprocess] Exporting point cloud...")
    await engine._run_export(output_db_abs, str(session.absolute()))
    
    print(f"[Reprocess] Parsing results...")
    from slam_engines.rtabmap.database_parser import DatabaseParser
    parser = DatabaseParser()
    result = await parser.parse_database(output_db)
    
    print(f"\n=== Reprocess Complete ===")
    print(f"Nodes (keyframes): {result['num_keyframes']}")
    print(f"Map points: {result['num_map_points']:,}")
    print(f"Loop closures: {result.get('loop_closures', 0)}")
    
    import sqlite3
    conn = sqlite3.connect(output_db)
    cursor = conn.execute("SELECT COUNT(*) FROM Word")
    words = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM Feature")
    features = cursor.fetchone()[0]
    conn.close()
    
    print(f"\n=== Relocalization Data ===")
    print(f"Visual words: {words:,}")
    print(f"Features: {features:,}")
    print(f"Features per keyframe: {features/result['num_keyframes'] if result['num_keyframes'] > 0 else 0:.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reprocess_session.py <session_path>")
        print("\nExample:")
        print("  python reprocess_session.py data/sessions/session_20260210_031736_42ea6383")
        sys.exit(1)
    
    session_path = sys.argv[1]
    asyncio.run(reprocess_session(session_path))
