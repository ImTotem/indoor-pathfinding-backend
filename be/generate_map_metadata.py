#!/usr/bin/env python3
"""Generate metadata JSON for RTAB-Map database files without metadata."""

import sqlite3
import struct
import json
import math
from pathlib import Path
from datetime import datetime

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]
    
    trace = r11 + r22 + r33
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r32 - r23) * s
        qy = (r13 - r31) * s
        qz = (r21 - r12) * s
    elif r11 > r22 and r11 > r33:
        s = 2.0 * math.sqrt(1.0 + r11 - r22 - r33)
        qw = (r32 - r23) / s
        qx = 0.25 * s
        qy = (r12 + r21) / s
        qz = (r13 + r31) / s
    elif r22 > r33:
        s = 2.0 * math.sqrt(1.0 + r22 - r11 - r33)
        qw = (r13 - r31) / s
        qx = (r12 + r21) / s
        qy = 0.25 * s
        qz = (r23 + r32) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r33 - r11 - r22)
        qw = (r21 - r12) / s
        qx = (r13 + r31) / s
        qy = (r23 + r32) / s
        qz = 0.25 * s
    
    return [qx, qy, qz, qw]

def extract_keyframes_from_rtabmap_db(db_path):
    """Extract keyframes from RTAB-Map database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, pose FROM Node ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    
    keyframes = []
    
    for node_id, pose_blob in rows:
        if not pose_blob or len(pose_blob) != 48:
            continue
        
        data = struct.unpack('<12f', pose_blob)
        
        r11, r12, r13, tx = data[0], data[1], data[2], data[3]
        r21, r22, r23, ty = data[4], data[5], data[6], data[7]
        r31, r32, r33, tz = data[8], data[9], data[10], data[11]
        
        R = [[r11, r12, r13],
             [r21, r22, r23],
             [r31, r32, r33]]
        
        orientation = rotation_matrix_to_quaternion(R)
        
        # RTAB-Map coordinate system (observed from 4F->1F descending video):
        #   X: forward (0 -> 59, moving forward)
        #   Y: left (0 -> 23, moving left)
        #   Z: down (0 -> 12, descending floors - DOWN is POSITIVE!)
        #
        # Three.js coordinate system:
        #   X: right
        #   Y: up
        #   Z: forward (toward viewer)
        #
        # Mapping:
        #   RTAB X (forward) -> Three.js Z (forward)
        #   RTAB Y (left) -> Three.js -X (right = -left)
        #   RTAB Z (down) -> Three.js -Y (up = -down)
        keyframes.append({
            "id": node_id,
            "position": [-ty, -tz, tx],
            "orientation": orientation
        })
    
    return keyframes

def generate_metadata(db_path):
    """Generate metadata JSON for a RTAB-Map database."""
    db_path = Path(db_path)
    map_id = db_path.stem
    
    print(f"Processing {db_path}...")
    
    keyframes = extract_keyframes_from_rtabmap_db(db_path)
    
    created_at = datetime.fromtimestamp(db_path.stat().st_mtime).isoformat() + 'Z'
    
    metadata = {
        "map_id": map_id,
        "name": map_id,
        "created_at": created_at,
        "keyframe_count": len(keyframes),
        "keyframes": keyframes,
        "engine": "rtabmap"
    }
    
    meta_path = db_path.parent / f"{map_id}_meta.json"
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Generated {meta_path}")
    print(f"   Keyframes: {len(keyframes)}")
    
    return meta_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_map_metadata.py <rtabmap_db_path>")
        print("\nExample:")
        print("  python generate_map_metadata.py data/maps/260202-202240.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    generate_metadata(db_path)
