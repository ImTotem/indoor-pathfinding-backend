"""Debug script to compare iOS DB vs our DB structure."""

import sqlite3
import struct


def analyze_db(db_path: str, name: str):
    """Analyze RTAB-Map database structure."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Path: {db_path}")
    print(f"{'='*60}\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Basic stats
    cursor.execute("SELECT COUNT(*) FROM Node")
    node_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Data")
    data_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Data WHERE depth IS NOT NULL")
    depth_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Data WHERE image IS NOT NULL")
    image_count = cursor.fetchone()[0]
    
    print(f"Nodes: {node_count}")
    print(f"Data rows: {data_count}")
    print(f"Rows with image: {image_count}")
    print(f"Rows with depth: {depth_count}")
    print(f"Depth coverage: {depth_count/data_count*100:.1f}%\n")
    
    # 2. Check a sample Data row
    cursor.execute("""
        SELECT id, 
               LENGTH(image) as img_size, 
               LENGTH(depth) as depth_size,
               LENGTH(calibration) as calib_size
        FROM Data 
        LIMIT 1
    """)
    row = cursor.fetchone()
    if row:
        print(f"Sample Data row (id={row[0]}):")
        print(f"  Image size: {row[1] or 0} bytes")
        print(f"  Depth size: {row[2] or 0} bytes")
        print(f"  Calibration size: {row[3] or 0} bytes\n")
    
    # 3. Check calibration structure
    cursor.execute("SELECT id, calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1")
    row = cursor.fetchone()
    if row and row[1]:
        node_id, calib_blob = row
        print(f"Calibration blob for node {node_id}:")
        print(f"  Total size: {len(calib_blob)} bytes")
        
        # Try to parse (RTAB-Map CameraModel serialization)
        # Format varies, but typically starts with name length + name string
        try:
            offset = 0
            # Read string length (4 bytes)
            name_len = struct.unpack('i', calib_blob[offset:offset+4])[0]
            offset += 4
            
            # Read camera name
            if 0 < name_len < 1000:  # sanity check
                camera_name = calib_blob[offset:offset+name_len].decode('utf-8', errors='ignore')
                print(f"  Camera name: '{camera_name}' (length: {name_len})")
                offset += name_len
                
                # Next should be image size (width, height)
                if offset + 8 <= len(calib_blob):
                    img_width, img_height = struct.unpack('ii', calib_blob[offset:offset+8])
                    print(f"  Image dimensions: {img_width} x {img_height}")
                    offset += 8
                    
                    # Intrinsics K (fx, fy, cx, cy, ...)
                    if offset + 32 <= len(calib_blob):
                        fx, fy, cx, cy = struct.unpack('dddd', calib_blob[offset:offset+32])
                        print(f"  Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        except Exception as e:
            print(f"  Failed to parse: {e}")
    
    # 4. Check Parameters for SLAM mode
    cursor.execute("SELECT parameters FROM Info LIMIT 1")
    row = cursor.fetchone()
    if row:
        params = row[0]
        # Extract key parameters
        for key in ["RGBD/Enabled", "Odom/Strategy", "Kp/DetectorStrategy", "Mem/DepthAsMask"]:
            if key in params:
                # Find parameter value
                start = params.find(key + ":")
                if start != -1:
                    end = params.find(";", start)
                    value = params[start:end] if end != -1 else params[start:]
                    print(f"  {value}")
    
    conn.close()


if __name__ == "__main__":
    analyze_db("be/data/maps/260202-202240.db", "iOS RTAB-Map App")
    analyze_db("be/data/maps/map_session_20260209_191433_b716707e.db", "Our Flutter App")
