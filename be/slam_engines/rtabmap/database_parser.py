"""RTAB-Map database parser for extracting trajectory and map metadata."""

import math
import sqlite3
import struct
from typing import Dict, List, Optional
from pathlib import Path


class DatabaseParser:
    """Parser for RTAB-Map SQLite database (.db files)."""

    async def extract_point_cloud(self, db_path: str, max_points: int = 50000) -> List[List[float]]:
        """Extract 3D feature points transformed to world coordinates."""
        if not Path(db_path).exists() or max_points <= 0:
            return []

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT node_id, depth_x, depth_y, depth_z
                FROM Feature
                WHERE depth_x IS NOT NULL
                  AND depth_y IS NOT NULL
                  AND depth_z IS NOT NULL
                """
            )
            feature_rows = cursor.fetchall()

            if not feature_rows:
                return []

            cursor.execute("SELECT id, pose FROM Node")
            pose_rows = cursor.fetchall()

            pose_by_node: Dict[int, tuple] = {}
            for node_id, pose_blob in pose_rows:
                if not pose_blob or len(pose_blob) < 48:
                    continue
                pose_by_node[node_id] = struct.unpack('12f', pose_blob[:48])

            if not pose_by_node:
                return []

            if len(feature_rows) > max_points:
                step = len(feature_rows) / float(max_points)
                sampled_rows = [feature_rows[int(i * step)] for i in range(max_points)]
            else:
                sampled_rows = feature_rows

            points: List[List[float]] = []
            for node_id, depth_x, depth_y, depth_z in sampled_rows:
                pose = pose_by_node.get(node_id)
                if pose is None:
                    continue

                r11, r12, r13, tx = pose[0], pose[1], pose[2], pose[3]
                r21, r22, r23, ty = pose[4], pose[5], pose[6], pose[7]
                r31, r32, r33, tz = pose[8], pose[9], pose[10], pose[11]

                world_x = r11 * depth_x + r12 * depth_y + r13 * depth_z + tx
                world_y = r21 * depth_x + r22 * depth_y + r23 * depth_z + ty
                world_z = r31 * depth_x + r32 * depth_y + r33 * depth_z + tz

                points.append([world_x, world_y, world_z])

            return points
        except Exception as e:
            print(f"[RTAB-Map] Point cloud extraction error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    async def parse_database(self, db_path: str, keyframe_limit: int = 0) -> Dict:
        if not Path(db_path).exists():
            return {
                'num_keyframes': 0,
                'num_map_points': 0,
                'keyframes': [],
                'loop_closures': 0
            }
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM Node")
            num_nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Link WHERE type=2")
            num_loops = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Feature")
            num_features = cursor.fetchone()[0]
            
            keyframes = []
            query = "SELECT id, pose, stamp FROM Node ORDER BY id"
            if keyframe_limit > 0:
                query += f" LIMIT {keyframe_limit}"
            cursor.execute(query)
            
            for node_id, pose_blob, timestamp in cursor.fetchall():
                if pose_blob:
                    pose = self._parse_pose_blob(pose_blob)
                    if pose:
                        keyframes.append({
                            'id': node_id,
                            'timestamp': timestamp,
                            'position': pose['position'],
                            'orientation': pose['orientation']
                        })
            
            conn.close()
            
            return {
                'num_keyframes': num_nodes,
                'num_map_points': num_features,
                'keyframes': keyframes,
                'loop_closures': num_loops
            }
        except Exception as e:
            print(f"[RTAB-Map] DB parse error: {e}")
            return {
                'num_keyframes': 0,
                'num_map_points': 0,
                'keyframes': [],
                'loop_closures': 0,
                'error': str(e)
            }
    
    def _parse_pose_blob(self, blob: bytes) -> Optional[Dict]:
        try:
            if len(blob) < 48:
                return None
            
            values = struct.unpack('12f', blob[:48])
            
            r11, r12, r13 = values[0], values[1], values[2]
            r21, r22, r23 = values[4], values[5], values[6]
            r31, r32, r33 = values[8], values[9], values[10]
            tx, ty, tz = values[3], values[7], values[11]
            
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
            
            return {
                'position': [tx, ty, tz],
                'orientation': [qx, qy, qz, qw]
            }
        except Exception:
            return None
    
    async def export_trajectory(self, db_path: str, output_path: str):
        """
        Export trajectory to TUM format file.
        
        TUM format: timestamp tx ty tz qx qy qz qw
        
        Args:
            db_path: Path to rtabmap.db file
            output_path: Path to output trajectory file
        """
        result = await self.parse_database(db_path)
        
        with open(output_path, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for kf in result['keyframes']:
                pos = kf['position']
                ori = kf['orientation']
                f.write(
                    f"{kf['timestamp']} "
                    f"{pos[0]} {pos[1]} {pos[2]} "
                    f"{ori[0]} {ori[1]} {ori[2]} {ori[3]}\n"
                )
