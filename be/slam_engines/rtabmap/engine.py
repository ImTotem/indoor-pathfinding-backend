# slam_engines/rtabmap/engine.py
"""RTAB-Map SLAM engine implementation.

This module provides the RTABMapEngine class that implements visual SLAM
processing using RTAB-Map in either Docker or local execution mode.
"""

import subprocess
import os
import asyncio
from pathlib import Path
from typing import Dict, Callable, Optional
import sqlite3
import struct

from slam_interface.base import SLAMEngineBase
from config.settings import settings
from .config_generator import ConfigGenerator
from .database_parser import DatabaseParser
from .db_builder import build_database
from . import constants
import time
import shutil
from typing import List
import logging


class RTABMapEngine(SLAMEngineBase):
    """RTAB-Map SLAM engine.
    
    Supports two execution modes based on RTABMAP_PATH configuration:
    - Local mode: "/path/to/rtabmap" (local installation)
    - Docker mode: "docker://<container_name>" (e.g., "docker://rtabmap")
    
    The engine processes RGB image sequences and generates 3D maps with
    loop closure detection and graph optimization.
    
    Note on sensor data (IMU/gyro/magnetometer):
        IMU data is collected by the Flutter app and stored in chunk JSON files,
        but is NOT used by this SLAM engine. RTAB-Map's CLI (rtabmap-console)
        does not accept IMU input directly. Integration would require either:
        - ROS nodes (rtabmap_ros) with IMU topic subscription, or
        - Custom C++ code using rtabmap::SensorData API
        Both approaches require significant additional infrastructure.
        The visual-only SLAM is sufficient for indoor mapping with monocular camera.
        IMU data is preserved in storage for potential future use.
    """
    
    def __init__(self):
        """Initialize RTAB-Map engine with configuration."""
        self._parse_rtabmap_path()
        
        self.config_generator = ConfigGenerator()
        self.database_parser = DatabaseParser()
        
        # Validate environment
        self._validate_environment()
    
    def _parse_rtabmap_path(self):
        """Parse RTABMAP_PATH configuration to determine execution mode."""
        path = settings.RTABMAP_PATH if hasattr(settings, 'RTABMAP_PATH') else "docker://rtabmap"
        
        if path.startswith("docker://"):
            # Docker mode: "docker://rtabmap"
            self.use_docker = True
            self.container_name = path.replace("docker://", "")
            self.rtabmap_path = "/rtabmap"  # Container internal path
        else:
            # Local mode: "/path/to/rtabmap"
            self.use_docker = False
            self.container_name = None
            self.rtabmap_path = path
    
    def _validate_environment(self):
        """Validate execution environment (Docker container or local installation)."""
        if self.use_docker:
            self._check_docker_container()
        else:
            if not os.path.exists(self.rtabmap_path):
                print(f"[RTAB-Map] Warning: Path not found: {self.rtabmap_path}")
    
    def extract_intrinsics_from_db(self, db_path: str) -> dict:
        """Extract camera intrinsics from RTABMap database calibration BLOB.
        
        Parses the 164-byte calibration BLOB from the Data table to extract
        camera intrinsics (fx, fy, cx, cy) and image resolution (width, height).
        
        The BLOB format is defined by RTABMap's CameraModel::serialize():
        - Format: '<3i i 2i 4i i 9d 12f' (little-endian, 164 bytes total)
        - Header (44 bytes):
            - version[3]: 3 integers (12 bytes)
            - type: 1 integer (4 bytes)
            - imageSize[2]: width, height (8 bytes)
            - K_size, D_size, R_size, P_size: 4 integers (16 bytes)
            - localTransformSize: 1 integer (4 bytes)
        - K matrix (72 bytes): 9 doubles, row-major [K00, K01, K02, K10, K11, K12, K20, K21, K22]
        - LocalTransform (48 bytes): 12 floats (ignored for intrinsics)
        
        Intrinsics extraction:
        - fx = K[0] (K00)
        - fy = K[4] (K11)
        - cx = K[2] (K02)
        - cy = K[5] (K12)
        - width = imageSize[0]
        - height = imageSize[1]
        
        Args:
            db_path: Path to RTABMap .db file
            
        Returns:
            Dictionary containing:
                - fx: Focal length X (pixels)
                - fy: Focal length Y (pixels)
                - cx: Principal point X (pixels)
                - cy: Principal point Y (pixels)
                - width: Image width (pixels)
                - height: Image height (pixels)
                
        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If calibration BLOB is invalid or corrupted
        """
        logger = logging.getLogger(__name__)
        
        # Validate file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        logger.info(f"[RTAB-Map] Extracting camera intrinsics from: {db_path}")
        
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query first calibration BLOB
            cursor.execute("SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1")
            row = cursor.fetchone()
            
            conn.close()
            
            if not row or not row[0]:
                raise ValueError("No calibration data found in database")
            
            blob = row[0]
            
            # Validate BLOB size
            expected_size = struct.calcsize('<3i i 2i 4i i 9d 12f')
            if len(blob) != expected_size:
                raise ValueError(
                    f"Invalid calibration BLOB size: {len(blob)} bytes (expected {expected_size} bytes)"
                )
            
            # Parse BLOB
            # Format: '<3i i 2i 4i i 9d 12f' = 164 bytes
            # Breakdown:
            #   3i = version (12 bytes)
            #   i = type (4 bytes)
            #   2i = imageSize (8 bytes)
            #   4i = K_size, D_size, R_size, P_size (16 bytes)
            #   i = localTransformSize (4 bytes)
            #   9d = K matrix (72 bytes)
            #   12f = LocalTransform (48 bytes)
            data = struct.unpack('<3i i 2i 4i i 9d 12f', blob)
            
            # Extract header values
            # version = data[0:3]  # Not used for intrinsics
            # type_val = data[3]   # Not used for intrinsics
            width = data[4]
            height = data[5]
            # K_size = data[6]     # Not used for intrinsics
            # D_size = data[7]     # Not used for intrinsics
            # R_size = data[8]     # Not used for intrinsics
            # P_size = data[9]     # Not used for intrinsics
            # localTransformSize = data[10]  # Not used for intrinsics
            
            # Extract K matrix (indices 11-19)
            K = data[11:20]  # 9 doubles: [K00, K01, K02, K10, K11, K12, K20, K21, K22]
            
            # Extract intrinsics from K matrix
            fx = K[0]  # K00 (row 0, col 0)
            fy = K[4]  # K11 (row 1, col 1)
            cx = K[2]  # K02 (row 0, col 2)
            cy = K[5]  # K12 (row 1, col 2)
            
            # Validate extracted values
            if fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0 or width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid intrinsics values: fx={fx}, fy={fy}, cx={cx}, cy={cy}, "
                    f"width={width}, height={height}"
                )
            
            result = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height
            }
            
            logger.info(
                f"[RTAB-Map] Extracted intrinsics: "
                f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, "
                f"resolution={width}x{height}"
            )
            
            return result
            
        except sqlite3.Error as e:
            raise ValueError(f"Database error: {e}")
        except struct.error as e:
            raise ValueError(f"BLOB parsing error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to extract intrinsics: {e}")
    
    def scale_intrinsics(self, original: dict, new_width: int, new_height: int) -> dict:
        """Scale camera intrinsics for resolution change.
        
        Proportionally scales focal lengths and principal point when image resolution
        changes. This is essential for SLAM when input frames have different resolution
        than the original calibration.
        
        The scaling formula is based on the fact that intrinsics are in pixel units:
        - scale_x = new_width / original_width
        - scale_y = new_height / original_height
        - fx_new = fx_old × scale_x
        - fy_new = fy_old × scale_y
        - cx_new = cx_old × scale_x
        - cy_new = cy_old × scale_y
        
        Args:
            original: Original intrinsics dict from extract_intrinsics_from_db()
                     Must contain: fx, fy, cx, cy, width, height
            new_width: Target image width (pixels)
            new_height: Target image height (pixels)
            
        Returns:
            New intrinsics dict with scaled values:
                - fx: Scaled focal length X
                - fy: Scaled focal length Y
                - cx: Scaled principal point X
                - cy: Scaled principal point Y
                - width: new_width
                - height: new_height
                
        Raises:
            ValueError: If scaling factor is extreme (>5x or <0.2x)
        """
        logger = logging.getLogger(__name__)
        
        old_width = original['width']
        old_height = original['height']
        
        # Check if scaling needed
        if old_width == new_width and old_height == new_height:
            logger.info("[RTAB-Map] No scaling needed (resolution matches)")
            return original
        
        # Calculate scale factors
        scale_x = new_width / old_width
        scale_y = new_height / old_height
        
        # Validate extreme scaling (prevent unrealistic transformations)
        if scale_x > 5.0 or scale_y > 5.0 or scale_x < 0.2 or scale_y < 0.2:
            raise ValueError(
                f"Extreme scaling detected: scale_x={scale_x:.3f}, scale_y={scale_y:.3f} "
                f"(max 5x, min 0.2x allowed)"
            )
        
        # Scale intrinsics
        fx_new = original['fx'] * scale_x
        fy_new = original['fy'] * scale_y
        cx_new = original['cx'] * scale_x
        cy_new = original['cy'] * scale_y
        
        logger.info(
            f"[RTAB-Map] Scaling intrinsics: {old_width}x{old_height} → {new_width}x{new_height}"
        )
        logger.info(
            f"[RTAB-Map] Scale factors: x={scale_x:.3f}, y={scale_y:.3f}"
        )
        logger.info(
            f"[RTAB-Map] Scaled values: "
            f"fx={fx_new:.2f}, fy={fy_new:.2f}, cx={cx_new:.2f}, cy={cy_new:.2f}"
        )
        
        return {
            'fx': fx_new,
            'fy': fy_new,
            'cx': cx_new,
            'cy': cy_new,
            'width': new_width,
            'height': new_height
        }
    
    def extract_params_from_db(self, db_path: str) -> dict:
        """Extract SLAM parameters from database Info table.
        
        Parses the semicolon-separated parameter string from Info.parameters
        and returns relevant SLAM configuration needed for relocalization.
        
        Args:
            db_path: Path to RTAB-Map .db file
            
        Returns:
            Dictionary with SLAM parameters (e.g., BRIEF/Bytes, Kp/DetectorStrategy)
        """
        import sqlite3
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT parameters FROM Info LIMIT 1")
            row = cursor.fetchone()
            
            if not row or not row[0]:
                return {}
            
            param_str = row[0]
            params = {}
            
            for pair in param_str.split(';'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    params[key] = value
            
            conn.close()
            
            return params
            
        except Exception as e:
            raise ValueError(f"Failed to extract parameters: {e}")
    
    def _check_docker_container(self):
        """Check if Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self.container_name}"],
                capture_output=True, text=True, timeout=5
            )
            if not result.stdout.strip():
                print(f"[RTAB-Map] ⚠️  Container '{self.container_name}' is not running")
                print(f"[RTAB-Map] Start it with: cd slam_engines/rtabmap/docker && ./run.sh")
            else:
                print(f"[RTAB-Map] ✅ Docker container '{self.container_name}' is running")
        except Exception as e:
            print(f"[RTAB-Map] ⚠️  Docker check failed: {e}")
    
    def _to_container_path(self, host_path: str) -> str:
        """Convert host path to container path (Docker mode only).
        
        Args:
            host_path: Path on the host machine
            
        Returns:
            Container path if Docker mode, otherwise unchanged host path
        """
        if not self.use_docker:
            return host_path
        
        data_dir = str(settings.DATA_DIR)
        if host_path.startswith(data_dir):
            return host_path.replace(data_dir, "/data")
        return host_path
    
    async def process(
        self, 
        session_id: str, 
        frames_data: Dict, 
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Process image sequence to generate 3D map using RTAB-Map.
        
        Pipeline:
          1. Extract camera intrinsics from poses
          2. Build RTAB-Map DB directly (images + depth + calibration + timestamps)
          3. Run rtabmap-reprocess with odometry to compute poses and features
          4. Parse output DB for metadata
        """
        
        print(f"\n{'='*50}")
        print(f"[RTAB-Map] Processing started: {session_id}")
        print(f"[RTAB-Map] Mode: {'Docker' if self.use_docker else 'Local'}")
        print(f"{'='*50}\n")
        
        session_path = frames_data['session_path']
        poses = frames_data['poses']
        
        try:
            if progress_callback:
                await progress_callback(5)

            camera_intrinsics = self._extract_camera_intrinsics(poses)
            if not camera_intrinsics:
                raise ValueError("No camera intrinsics found in poses")
            
            if progress_callback:
                await progress_callback(10)

            input_db = build_database(
                session_path, 
                camera_intrinsics,
                slam_params=constants.DEFAULT_PARAMS
            )
            
            if progress_callback:
                await progress_callback(30)

            output_db = f"{session_path}/{constants.DATABASE_FILENAME}"
            await self._run_reprocess(input_db, output_db, progress_callback)
            
            if progress_callback:
                await progress_callback(80)

            await self._run_export(output_db, session_path)

            if progress_callback:
                await progress_callback(90)
            
            parsed = await self.database_parser.parse_database(output_db)
            map_binary = await self._load_map_file(output_db)
            
            if progress_callback:
                await progress_callback(100)
            
            print(f"[RTAB-Map] Processing completed\n")
            
            return {
                "binary": map_binary,
                "metadata": {
                    "session_id": session_id,
                    "session_path": session_path,
                    "num_keyframes": parsed['num_keyframes'],
                    "num_map_points": parsed['num_map_points'],
                    "keyframes": parsed.get('keyframes', []),
                    "loop_closures": parsed.get('loop_closures', 0),
                    "slam_engine": "RTAB-Map",
                    "status": "completed",
                    "camera_intrinsics": camera_intrinsics,
                }
            }
            
        except Exception as e:
            print(f"[RTAB-Map] Processing failed: {e}")
            raise
    
    async def _run_reprocess(
        self,
        input_db: str,
        output_db: str,
        progress_callback: Optional[Callable] = None
    ):
        """Run rtabmap-reprocess to compute odometry, features, and optimization."""
        params = dict(constants.DEFAULT_PARAMS)
        params["RGBD/Enabled"] = "true"
        
        cli_args = self.config_generator.params_to_cli_args(params)
        
        if self.use_docker:
            assert self.container_name is not None
            c_input = self._to_container_path(input_db)
            c_output = self._to_container_path(output_db)
            
            reprocess_cmd = ["rtabmap-reprocess", "-odom"] + cli_args + [c_input, c_output]
            bash_cmd = " ".join(reprocess_cmd)
            cmd = ["docker", "exec", self.container_name, "bash", "-c", bash_cmd]
        else:
            cmd = ["rtabmap-reprocess", "-odom"] + cli_args + [input_db, output_db]
        
        print(f"[RTAB-Map] Running reprocess: {'Docker' if self.use_docker else 'Local'}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            total_images = 0
            processed = 0
            
            async def monitor_output():
                nonlocal total_images, processed
                if process.stdout is None:
                    return
                async for line in process.stdout:
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        print(f"[RTAB-Map] {line_str}")
                        if "Processing image" in line_str or "Odom" in line_str:
                            processed += 1
                            if progress_callback and total_images > 0:
                                pct = 30 + (processed / total_images * 55)
                                await progress_callback(min(pct, 85))
            
            monitor_task = asyncio.create_task(monitor_output())
            
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=constants.SLAM_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                if self.use_docker and self.container_name:
                    subprocess.run([
                        "docker", "exec", self.container_name,
                        "pkill", "-9", "-f", "rtabmap-reprocess"
                    ], capture_output=True)
                else:
                    process.kill()
                raise Exception(f"rtabmap-reprocess timeout ({constants.SLAM_TIMEOUT_SECONDS}s)")
            
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            print(f"[RTAB-Map] reprocess completed (code: {process.returncode})")
            
            if process.returncode != 0:
                stderr_data = await process.stderr.read() if process.stderr else b""
                error_msg = stderr_data.decode('utf-8', errors='ignore')
                print(f"[RTAB-Map] Error: {error_msg}")
                raise Exception(f"rtabmap-reprocess failed (code: {process.returncode})")
            
            if not os.path.exists(output_db):
                raise Exception(f"Output database not created: {output_db}")
                
        except FileNotFoundError:
            raise Exception("rtabmap-reprocess not found")
        except Exception as e:
            if "timeout" in str(e).lower():
                raise
            raise Exception(f"rtabmap-reprocess failed: {e}")
    

    
    async def _run_export(self, db_path: str, session_path: str):
        """Run rtabmap-export to generate a dense colored PLY point cloud."""
        if self.use_docker:
            assert self.container_name is not None
            c_db = self._to_container_path(db_path)
            c_out = self._to_container_path(session_path)
            cmd = [
                "docker", "exec", self.container_name, "bash", "-c",
                f"rtabmap-export --cloud --opt 3 --output_dir {c_out}/ {c_db}"
            ]
        else:
            cmd = [
                "rtabmap-export", "--cloud", "--opt", "3",
                "--output_dir", f"{session_path}/", db_path
            ]

        print("[RTAB-Map] Exporting dense point cloud...")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=120
            )

            if process.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore")
                print(f"[RTAB-Map] Export warning (code {process.returncode}): {err}")
            else:
                out = stdout.decode("utf-8", errors="ignore")
                print(f"[RTAB-Map] Export done: {out.strip().splitlines()[-1] if out.strip() else 'ok'}")

        except asyncio.TimeoutError:
            print("[RTAB-Map] Export timeout (120s), skipping PLY generation")
        except Exception as e:
            print(f"[RTAB-Map] Export failed: {e}")

    def _extract_camera_intrinsics(self, poses: list) -> Optional[Dict]:
        """Extract camera intrinsics from first pose.
        
        Args:
            poses: List of pose dictionaries
            
        Returns:
            Camera intrinsics dict or None if not available
        """
        if not poses:
            return None
        
        first_pose = poses[0]
        return first_pose.get('camera_intrinsics')
    

    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH.
        
        Args:
            command: Command name to check
            
        Returns:
            True if command exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", command],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def _wait_for_database(self, session_path: str, timeout: int = 30):
        """Wait for database file to be created.
        
        Args:
            session_path: Session directory path
            timeout: Maximum wait time in seconds
        """
        db_path = Path(session_path) / constants.DATABASE_FILENAME
        
        for _ in range(timeout * 10):  # Check every 0.1s
            if db_path.exists() and db_path.stat().st_size > 0:
                print(f"[RTAB-Map] Database file created: {db_path}")
                return
            await asyncio.sleep(0.1)
        
        raise Exception(f"Database file not created after {timeout}s")
    
    async def _load_map_file(self, db_path: str) -> bytes:
        """Load RTAB-Map database file.
        
        Args:
            db_path: Path to rtabmap.db file
            
        Returns:
            Database file contents as bytes
        """
        if not os.path.exists(db_path):
            print(f"[RTAB-Map] Warning: Database file not found: {db_path}")
            return b""
        
        print(f"[RTAB-Map] Loading database: {db_path}")
        with open(db_path, 'rb') as f:
            data = f.read()
        
        print(f"[RTAB-Map] Database loaded: {len(data)} bytes")
        return data
    
    async def localize(self, map_id: str, images: List[bytes], intrinsics: Optional[Dict] = None, initial_pose: Optional[Dict] = None) -> dict:
        """Localize current position using 1-5 query images against existing map.
        
        Args:
            map_id: ID of the RTABMap database to localize against
            images: List of 1-5 image bytes (JPEG/PNG)
            intrinsics: Optional camera intrinsics (auto-extracted if None)
            initial_pose: Optional initial pose estimate (not used)
        
        Returns:
            dict: {
                "pose": {"x", "y", "z", "qx", "qy", "qz", "qw"},
                "confidence": float,
                "map_id": str,
                "num_matches": int
            }
        
        Raises:
            FileNotFoundError: Map database not found
            ValueError: Invalid number of images or processing failed
            TimeoutError: RTABMap processing timeout (30s)
        """
        # 1. Validate map exists
        map_path = settings.MAPS_DIR / f"{map_id}.db"
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_id}")
        
        # 2. Create temporary session directory for query images
        session_id = f"reloc_{map_id}_{int(time.time())}"
        session_dir = settings.SESSIONS_DIR / session_id
        images_dir = session_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 3. Save query images
            for i, img_bytes in enumerate(images, 1):
                img_path = images_dir / f"query_{i}.jpg"
                img_path.write_bytes(img_bytes)
            
            # 4. Create images.txt (RTABMap input format)
            images_txt = session_dir / "images.txt"
            with open(images_txt, 'w') as f:
                f.write("# timestamp filename\n")
                for i in range(1, len(images) + 1):
                    f.write(f"{i} images/query_{i}.jpg\n")
            
            # 5. Set relocalization parameters
            db_params = self.extract_params_from_db(str(map_path))
            
            reloc_params = constants.DEFAULT_PARAMS.copy()
            
            if 'BRIEF/Bytes' in db_params:
                reloc_params['BRIEF/Bytes'] = db_params['BRIEF/Bytes']
                print(f"[RTAB-Map] Using map's BRIEF descriptor size: {db_params['BRIEF/Bytes']} bytes")
            
            if 'Kp/DetectorStrategy' in db_params:
                reloc_params['Kp/DetectorStrategy'] = db_params['Kp/DetectorStrategy']
            
            reloc_params["Mem/IncrementalMemory"] = "false"
            reloc_params["Mem/InitWMWithAllNodes"] = "true"
            cli_args = self.config_generator.params_to_cli_args(reloc_params)
            
            # 6. Invoke rtabmap-console (subprocess CLI)
            # Correct syntax: rtabmap-console [params] -input existing_map.db images_directory
            if self.use_docker:
                c_map_path = self._to_container_path(str(map_path))
                c_images_dir = self._to_container_path(str(images_dir))
                
                rtabmap_cmd = [
                    "rtabmap-console"
                ] + cli_args + [
                    "-input", c_map_path,
                    c_images_dir
                ]
                
                bash_cmd = " ".join(rtabmap_cmd)
                cmd = ["docker", "exec", self.container_name, "bash", "-c", bash_cmd]
            else:
                # Local mode invocation
                cmd = [
                    "rtabmap-console"
                ] + cli_args + [
                    "-input", str(map_path),
                    str(images_dir)
                ]
            
            # 7. Execute with 30-second timeout
            print(f"[RTAB-Map] Relocalization starting for map {map_id}...")
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=30  # CRITICAL: 30s not 600s
                    )
                except asyncio.TimeoutError:
                    # Kill process on timeout
                    if self.use_docker and self.container_name:
                        subprocess.run([
                            "docker", "exec", self.container_name,
                            "pkill", "-9", "-f", "rtabmap-console"
                        ], capture_output=True)
                    else:
                        process.kill()
                    raise TimeoutError("RTABMap relocalization timeout (exceeded 30 seconds)")
                
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    print(f"[RTAB-Map] Relocalization failed: {error_msg}")
                    raise ValueError(f"RTABMap processing failed: {error_msg}")
                
                # 8. Parse output for matched node ID
                stdout_str = stdout.decode('utf-8', errors='ignore')
                print(f"[RTAB-Map] Relocalization output:\n{stdout_str}")
                
                # 8a. Parse iteration lines for matched node ID
                import re
                matched_node_id = None
                best_hypothesis = -999.0
                
                loop_pattern = re.compile(r'iteration\(\d+\)\s+loop\((\d+)\)\s+hyp\(([-0-9.]+)\)')
                high_pattern = re.compile(r'iteration\(\d+\)\s+high\((\d+)\)\s+hyp\(([-0-9.]+)\)')
                
                for line in stdout_str.split('\n'):
                    loop_match = loop_pattern.search(line)
                    if loop_match:
                        node_id = int(loop_match.group(1))
                        hypothesis = float(loop_match.group(2))
                        if hypothesis > best_hypothesis:
                            best_hypothesis = hypothesis
                            matched_node_id = node_id
                        print(f"[RTAB-Map] Loop closure accepted: node={node_id}, hyp={hypothesis}")
                        continue
                    
                    high_match = high_pattern.search(line)
                    if high_match and matched_node_id is None:
                        node_id = int(high_match.group(1))
                        hypothesis = float(high_match.group(2))
                        if hypothesis > best_hypothesis:
                            best_hypothesis = hypothesis
                            matched_node_id = node_id
                        print(f"[RTAB-Map] Best hypothesis: node={node_id}, hyp={hypothesis}")
                
                if matched_node_id is None:
                    raise ValueError("No loop closure or hypothesis found - relocalization failed")
                
                print(f"[RTAB-Map] Final matched node: {matched_node_id} (hypothesis={best_hypothesis})")
                
                # 8b. Extract pose from matched node
                pose = self._extract_node_pose(map_path, matched_node_id)
                if pose is None:
                    raise ValueError(f"Failed to extract pose for node {matched_node_id}")
                
                # 8c. Calculate confidence from hypothesis value
                confidence = min(0.9, max(0.1, (best_hypothesis + 1.0) / 2.0))
                
                # 9. Return result matching API contract
                return {
                    "pose": {
                        "x": pose[0],
                        "y": pose[1],
                        "z": pose[2],
                        "qx": pose[3],
                        "qy": pose[4],
                        "qz": pose[5],
                        "qw": pose[6]
                    },
                    "confidence": confidence,
                    "map_id": map_id,
                    "matched_node_id": matched_node_id
                }
            
            except FileNotFoundError:
                raise Exception("rtabmap-console not found - is RTABMap installed?")
            except Exception as e:
                if isinstance(e, (FileNotFoundError, ValueError, TimeoutError)):
                    raise
                raise Exception(f"RTABMap relocalization failed: {e}")
        
        finally:
            # 10. Clean up temporary files
            if session_dir.exists():
                shutil.rmtree(session_dir)
                print(f"[RTAB-Map] Cleaned up temporary session: {session_id}")
    
    def _parse_localization_output(self, stdout: str) -> tuple:
        """Parse RTABMap stdout for pose and confidence.
        
        Args:
            stdout: RTABMap console output
            
        Returns:
            tuple: (pose, confidence, num_matches)
                pose: [x, y, z, qx, qy, qz, qw]
                confidence: float (0.0-1.0)
                num_matches: int
        """
        import re

        float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
        pose_patterns = [
            rf"Localized\s*:?\s*(?P<x>{float_pattern})\s+(?P<y>{float_pattern})\s+(?P<z>{float_pattern})\s+(?P<qx>{float_pattern})\s+(?P<qy>{float_pattern})\s+(?P<qz>{float_pattern})\s+(?P<qw>{float_pattern})",
            rf"Pose\s*:?\s*(?P<x>{float_pattern})\s+(?P<y>{float_pattern})\s+(?P<z>{float_pattern})\s+(?P<qx>{float_pattern})\s+(?P<qy>{float_pattern})\s+(?P<qz>{float_pattern})\s+(?P<qw>{float_pattern})",
            rf"t\s*=\s*\[\s*(?P<x>{float_pattern})[\s,]+(?P<y>{float_pattern})[\s,]+(?P<z>{float_pattern})\s*\]\s*q\s*=\s*\[\s*(?P<qx>{float_pattern})[\s,]+(?P<qy>{float_pattern})[\s,]+(?P<qz>{float_pattern})[\s,]+(?P<qw>{float_pattern})\s*\]",
            rf"x\s*=\s*(?P<x>{float_pattern})\s*y\s*=\s*(?P<y>{float_pattern})\s*z\s*=\s*(?P<z>{float_pattern})\s*qx\s*=\s*(?P<qx>{float_pattern})\s*qy\s*=\s*(?P<qy>{float_pattern})\s*qz\s*=\s*(?P<qz>{float_pattern})\s*qw\s*=\s*(?P<qw>{float_pattern})",
        ]

        def parse_pose_from_line(line: str) -> Optional[List[float]]:
            for pattern in pose_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return [
                        float(match.group("x")),
                        float(match.group("y")),
                        float(match.group("z")),
                        float(match.group("qx")),
                        float(match.group("qy")),
                        float(match.group("qz")),
                        float(match.group("qw")),
                    ]

            if any(token in line.lower() for token in ("localized", "localization", "pose", "transform")):
                numbers = [float(value) for value in re.findall(float_pattern, line)]
                if len(numbers) >= 7:
                    return numbers[-7:]
            return None

        pose = None
        confidence = 0.0
        num_matches = 0
        num_inliers = 0
        loop_accepted = False

        lines = stdout.split("\n")
        for line in lines:
            lower = line.lower()

            if pose is None:
                pose = parse_pose_from_line(line)

            if "loop" in lower and "accepted" in lower:
                loop_accepted = True

            match = re.search(r"\bmatches\b\s*[:=]\s*(\d+)", lower)
            if match:
                num_matches = max(num_matches, int(match.group(1)))

            match = re.search(r"\binliers\b\s*[:=]\s*(\d+)(?:\s*/\s*(\d+))?", lower)
            if match:
                num_inliers = max(num_inliers, int(match.group(1)))
                if match.group(2):
                    num_matches = max(num_matches, int(match.group(2)))

        if pose is None:
            raise ValueError("pose not found in RTABMap output")

        if num_matches > 0 and num_inliers > 0:
            ratio = num_inliers / max(num_matches, 1)
            confidence = max(confidence, min(1.0, 0.2 + ratio * 0.8))
        elif num_matches > 0:
            confidence = max(confidence, min(1.0, num_matches / 60.0))
        elif num_inliers > 0:
            confidence = max(confidence, min(1.0, num_inliers / 60.0))

        if loop_accepted:
            confidence = max(confidence, 0.7)

        if confidence == 0.0:
            confidence = 0.5

        if confidence < 0.5:
            raise ValueError("insufficient feature matches for reliable relocalization")

        return pose, confidence, num_matches
    
    def _extract_node_pose(self, db_path: Path, node_id: int) -> Optional[List[float]]:
        import struct
        import math
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, pose FROM Node WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            conn.close()
            
            if not row or not row[1]:
                print(f"[RTAB-Map] Node {node_id} not found or has no pose")
                return None
            
            pose_blob = row[1]
            
            if len(pose_blob) != 48:
                print(f"[RTAB-Map] Unexpected pose blob size: {len(pose_blob)} (expected 48)")
                return None
            
            data = struct.unpack('<12f', pose_blob)
            
            r11, r12, r13, tx = data[0], data[1], data[2], data[3]
            r21, r22, r23, ty = data[4], data[5], data[6], data[7]
            r31, r32, r33, tz = data[8], data[9], data[10], data[11]
            
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
            
            return [tx, ty, tz, qx, qy, qz, qw]
            
        except Exception as e:
            print(f"[RTAB-Map] Failed to extract pose from database: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_map(self, map_data: Dict, map_id: str, base_dir: Path) -> Path:
        """Save RTABMap database and PLY point cloud to maps directory."""
        map_path = base_dir / f"{map_id}.db"

        with open(map_path, "wb") as f:
            f.write(map_data["binary"])

        print(f"[RTAB-Map] Map saved: {map_path} ({len(map_data['binary'])} bytes)")

        session_path = map_data.get("metadata", {}).get("session_path")
        if session_path:
            ply_src = Path(session_path) / "rtabmap_cloud.ply"
            if ply_src.exists():
                ply_dst = base_dir / f"{map_id}.ply"
                shutil.copy2(str(ply_src), str(ply_dst))
                print(f"[RTAB-Map] PLY saved: {ply_dst}")

        return map_path
    
    def load_map(self, map_id: str, base_dir: Path) -> bytes:
        """Load RTABMap database from .db file.
        
        Args:
            map_id: Unique map identifier
            base_dir: Base directory containing maps
            
        Returns:
            Map file contents as bytes
            
        Raises:
            FileNotFoundError: If map file doesn't exist
        """
        map_path = base_dir / f"{map_id}.db"
        
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")
        
        print(f"[RTAB-Map] Loading map: {map_path}")
        with open(map_path, "rb") as f:
            data = f.read()
        
        print(f"[RTAB-Map] Map loaded: {len(data)} bytes")
        return data
