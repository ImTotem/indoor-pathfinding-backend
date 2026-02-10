# routes/viewer.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from typing import Optional
from pathlib import Path
import json
import os

from config.settings import settings
from slam_engines.rtabmap.database_parser import DatabaseParser

router = APIRouter(prefix="/api/viewer", tags=["viewer"])


def _resolve_map_db_path(map_id: str) -> Optional[Path]:
    maps_dir = settings.MAPS_DIR

    direct_candidates = [
        maps_dir / f"{map_id}.db",
        maps_dir / map_id,
        maps_dir / f"map_{map_id}.db",
    ]

    for candidate in direct_candidates:
        if candidate.exists() and candidate.suffix == ".db":
            return candidate

    meta_candidates = [
        maps_dir / f"{map_id}_meta.json",
        maps_dir / f"map_{map_id}_meta.json",
    ]

    for meta_path in meta_candidates:
        if meta_path.exists():
            db_stem = meta_path.name.replace("_meta.json", "")
            db_path = maps_dir / f"{db_stem}.db"
            if db_path.exists():
                return db_path

    return None


def _resolve_ply_path(map_id: str) -> Optional[Path]:
    maps_dir = settings.MAPS_DIR
    for candidate in [
        maps_dir / f"{map_id}.ply",
        maps_dir / f"map_{map_id}.ply",
    ]:
        if candidate.exists():
            return candidate

    sessions_dir = settings.DATA_DIR / "sessions"
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            ply = session_dir / "rtabmap_cloud.ply"
            if ply.exists():
                meta_id = f"map_{session_dir.name}"
                if meta_id == map_id or session_dir.name in map_id:
                    return ply

    return None


@router.get("/map/{map_id}/ply")
async def get_map_ply(map_id: str):
    ply_path = _resolve_ply_path(map_id)
    if not ply_path:
        raise HTTPException(status_code=404, detail=f"PLY file not found for map: {map_id}")

    return FileResponse(
        path=str(ply_path),
        media_type="application/octet-stream",
        filename=f"{map_id}.ply",
    )


@router.get("/map/{map_id}/points")
async def get_map_points(
    map_id: str,
    max_points: int = Query(50000, ge=1, le=200000),
):
    db_path = _resolve_map_db_path(map_id)
    if not db_path:
        raise HTTPException(status_code=404, detail=f"Map database not found: {map_id}")

    parser = DatabaseParser()
    points = await parser.extract_point_cloud(str(db_path), max_points=max_points)
    return {"points": points, "count": len(points)}

@router.get("/map/{map_id}", response_class=HTMLResponse)
async def view_map(map_id: str, pose: Optional[str] = Query(None, description="Camera pose as 'x,y,z'")):
    db_path = _resolve_map_db_path(map_id)
    if not db_path:
        raise HTTPException(status_code=404, detail=f"Map database not found: {map_id}")

    pose_data = None
    if pose:
        try:
            x, y, z = map(float, pose.split(','))
            pose_data = {"x": x, "y": y, "z": z}
        except (ValueError, AttributeError):
            pose_data = None

    try:
        parser = DatabaseParser()
        parsed = await parser.parse_database(str(db_path))
        keyframes = parsed['keyframes']
        num_points = parsed['num_map_points']
        loop_closures = parsed.get('loop_closures', 0)

        travelled = 0.0
        if len(keyframes) > 1:
            for i in range(1, len(keyframes)):
                p0 = keyframes[i - 1]['position']
                p1 = keyframes[i]['position']
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                dz = p1[2] - p0[2]
                travelled += (dx * dx + dy * dy + dz * dz) ** 0.5

        center_x, center_y, center_z = 0, 0, 0
        min_y = 0
        max_y = 0
        max_span = 5

        if keyframes:
            positions = [kf['position'] for kf in keyframes]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            center_x = (min(xs) + max(xs)) / 2
            center_y = (min(ys) + max(ys)) / 2
            center_z = (min(zs) + max(zs)) / 2
            min_y = min(ys)
            max_y = max(ys)
            max_span = max(max(xs) - min(xs), max(zs) - min(zs), 2)

        ply_available = _resolve_ply_path(map_id) is not None
        db_size_mb = os.path.getsize(str(db_path)) / (1024 * 1024)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Indoor Map Viewer - {map_id}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #1a1a1a; }}
        canvas {{ display: block; }}
        #stats {{
            position: absolute;
            top: 12px;
            left: 12px;
            color: #e0e0e0;
            background: rgba(0,0,0,0.82);
            padding: 14px 18px;
            border-radius: 6px;
            font: 13px/1.7 'Consolas','Monaco','Courier New',monospace;
            pointer-events: none;
            min-width: 260px;
            z-index: 10;
        }}
        #stats .label {{ color: #999; }}
        #stats .value {{ color: #fff; }}
        #toggles {{
            position: absolute;
            top: 12px;
            right: 12px;
            color: #ccc;
            background: rgba(0,0,0,0.82);
            padding: 10px 14px;
            border-radius: 6px;
            font: 13px/2 sans-serif;
            z-index: 10;
        }}
        #toggles label {{ display: block; cursor: pointer; user-select: none; }}
        #toggles label:hover {{ color: #4af; }}
        #controls {{
            position: absolute;
            bottom: 12px;
            left: 12px;
            color: #888;
            background: rgba(0,0,0,0.7);
            padding: 8px 12px;
            border-radius: 6px;
            font: 11px/1.5 sans-serif;
            pointer-events: none;
        }}
        #loading {{
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%,-50%);
            color: #fff;
            font: 16px sans-serif;
            z-index: 20;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="loading">포인트 클라우드 로딩 중...</div>
    <div id="stats">
        <span class="label">Nodes (WM):</span> <span class="value">{len(keyframes)}</span><br>
        <span class="label">Database (MB):</span> <span class="value">{db_size_mb:.1f}</span><br>
        <span class="label">Features:</span> <span class="value">{num_points}</span><br>
        <span class="label" id="points-label">Points:</span> <span class="value" id="points-count">loading...</span><br>
        <span class="label">Loop closures:</span> <span class="value">{loop_closures}</span><br>
        <span class="label">Travelled distance:</span> <span class="value">{travelled:.2f} m</span><br>
        <span class="label">Pose (x,y,z):</span> <span class="value" id="cam-pose">0.00 0.00 0.00</span><br>
        <span class="label">FPS:</span> <span class="value" id="fps-counter">-</span>
    </div>
    <div id="toggles">
        <label><input type="checkbox" checked data-layer="pointCloud"> 포인트 클라우드</label>
        <label><input type="checkbox" checked data-layer="trajectory"> 경로</label>
        <label><input type="checkbox" checked data-layer="keyframes"> 키프레임</label>
        <label><input type="checkbox" checked data-layer="grid"> 그리드</label>
        <label><input type="checkbox" data-layer="axes"> 축</label>
    </div>
    <div id="controls">
        좌클릭: 회전 &nbsp;|&nbsp; 우클릭: 이동 &nbsp;|&nbsp; WASD: 전후좌우 &nbsp;|&nbsp; Q/E: 상하 &nbsp;|&nbsp; 스크롤: 전후 &nbsp;|&nbsp; R: 리셋
    </div>
    <script>
        const rt2t = (x, y, z) => [x, y, z];

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);

        const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 500);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(innerWidth, innerHeight);
        renderer.setPixelRatio(devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        const initRaw = [{center_x}, {center_y}, {center_z}];
        const initT = rt2t(...initRaw);
        const camOffsetRaw = [{max_span * 0.8}, {max_span * 0.6}, {max_span * 0.8}];
        const camPosT = [initT[0] + camOffsetRaw[0], initT[1] + camOffsetRaw[1], initT[2] + camOffsetRaw[2]];
        camera.position.set(...camPosT);
        camera.lookAt(...initT);

        const euler = new THREE.Euler(0, 0, 0, 'YXZ');
        euler.setFromQuaternion(camera.quaternion, 'YXZ');
        const keys = {{}};
        let isDragging = false, isPanning = false;
        let prevX = 0, prevY = 0;
        const sensitivity = 0.002;
        const moveSpeed = {max(max_span * 0.03, 0.05)};

        const groups = {{
            pointCloud: new THREE.Group(),
            trajectory: new THREE.Group(),
            keyframes: new THREE.Group(),
            grid: new THREE.Group(),
            axes: new THREE.Group()
        }};
        Object.values(groups).forEach(g => scene.add(g));

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const gridT = rt2t(0, {min_y} - 0.05, 0);
        const grid = new THREE.GridHelper(
            Math.ceil({max_span}) * 2,
            Math.ceil({max_span}) * 4,
            0x333333, 0x222222
        );
        grid.position.set(...gridT);
        groups.grid.add(grid);

        const axesT = rt2t({center_x}, {min_y} - 0.05, {center_z});
        const axes = new THREE.AxesHelper({max(max_span * 0.3, 0.5)});
        axes.position.set(...axesT);
        groups.axes.add(axes);
        groups.axes.visible = false;

        const kfData = {json.dumps(keyframes)};
        const pathPts = [];
        kfData.forEach(kf => {{
            const p = kf.position;
            pathPts.push(new THREE.Vector3(...rt2t(p[0], p[1], p[2])));
        }});

        if (pathPts.length > 1) {{
            const geo = new THREE.BufferGeometry().setFromPoints(pathPts);
            const mat = new THREE.LineBasicMaterial({{ color: 0x3388ff, linewidth: 2 }});
            groups.trajectory.add(new THREE.Line(geo, mat));

            const odomGeo = new THREE.BufferGeometry().setFromPoints(pathPts);
            const odomMat = new THREE.LineBasicMaterial({{ color: 0xff4444, linewidth: 1, opacity: 0.6, transparent: true }});
            groups.trajectory.add(new THREE.Line(odomGeo, odomMat));
        }}

        const camGeo = new THREE.SphereGeometry(0.015, 8, 8);
        const camMat = new THREE.MeshBasicMaterial({{ color: 0x00ff88 }});
        kfData.forEach(kf => {{
            const m = new THREE.Mesh(camGeo, camMat);
            m.position.set(...rt2t(kf.position[0], kf.position[1], kf.position[2]));
            groups.keyframes.add(m);
        }});

        const poseData = {json.dumps(pose_data)};
        if (poseData) {{
            const geo = new THREE.SphereGeometry(0.05, 16, 16);
            const mat = new THREE.MeshBasicMaterial({{ color: 0xff0000 }});
            const marker = new THREE.Mesh(geo, mat);
            marker.position.set(...rt2t(poseData.x, poseData.y, poseData.z));
            scene.add(marker);
        }}

        async function loadPLY() {{
            const el = document.getElementById('loading');
            const countEl = document.getElementById('points-count');
            try {{
                const resp = await fetch('/api/viewer/map/{map_id}/ply');
                if (!resp.ok) throw new Error('PLY not available');
                const buf = await resp.arrayBuffer();

                const loader = new THREE.PLYLoader();
                const geometry = loader.parse(buf);

                const pos = geometry.getAttribute('position');
                const n = pos.count;
                const newPos = new Float32Array(n * 3);
                for (let i = 0; i < n; i++) {{
                    const t = rt2t(pos.getX(i), pos.getY(i), pos.getZ(i));
                    newPos[i * 3] = t[0]; newPos[i * 3 + 1] = t[1]; newPos[i * 3 + 2] = t[2];
                }}
                geometry.setAttribute('position', new THREE.BufferAttribute(newPos, 3));

                if (geometry.getAttribute('normal')) {{
                    const norm = geometry.getAttribute('normal');
                    const newNorm = new Float32Array(n * 3);
                    for (let i = 0; i < n; i++) {{
                        const t = rt2t(norm.getX(i), norm.getY(i), norm.getZ(i));
                        newNorm[i * 3] = t[0]; newNorm[i * 3 + 1] = t[1]; newNorm[i * 3 + 2] = t[2];
                    }}
                    geometry.setAttribute('normal', new THREE.BufferAttribute(newNorm, 3));
                }}

                geometry.computeBoundingBox();

                const hasColor = !!geometry.getAttribute('color');
                const mat = new THREE.PointsMaterial({{
                    size: 2.0,
                    sizeAttenuation: false,
                    vertexColors: hasColor
                }});
                if (!hasColor) mat.color.set(0xaaaaaa);

                groups.pointCloud.add(new THREE.Points(geometry, mat));
                countEl.textContent = n.toLocaleString();
                el.style.display = 'none';
            }} catch (e) {{
                console.warn('PLY load failed, falling back to feature points:', e);
                el.textContent = '특징점 로딩 중...';
                await loadFallback();
                el.style.display = 'none';
            }}
        }}

        async function loadFallback() {{
            const countEl = document.getElementById('points-count');
            try {{
                const resp = await fetch('/api/viewer/map/{map_id}/points?max_points=100000');
                const data = await resp.json();
                const pts = data.points || [];
                if (!pts.length) {{ countEl.textContent = '0'; return; }}

                const positions = new Float32Array(pts.length * 3);
                const colors = new Float32Array(pts.length * 3);
                let minH = Infinity, maxH = -Infinity;
                pts.forEach(p => {{ if (p[1] < minH) minH = p[1]; if (p[1] > maxH) maxH = p[1]; }});
                const hR = maxH - minH || 1;
                pts.forEach((p, i) => {{
                    const t = rt2t(p[0], p[1], p[2]);
                    positions[i*3] = t[0]; positions[i*3+1] = t[1]; positions[i*3+2] = t[2];
                    const c = new THREE.Color(); c.setHSL(0.66 - ((p[1]-minH)/hR)*0.66, 0.9, 0.5);
                    colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
                }});
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                groups.pointCloud.add(new THREE.Points(geo, new THREE.PointsMaterial({{ size: 2, sizeAttenuation: false, vertexColors: true }})));
                countEl.textContent = pts.length.toLocaleString();
            }} catch (err) {{
                console.error(err);
                countEl.textContent = 'error';
            }}
        }}

        loadPLY();

        document.querySelectorAll('#toggles input').forEach(cb => {{
            cb.addEventListener('change', e => {{
                const g = groups[e.target.dataset.layer];
                if (g) g.visible = e.target.checked;
            }});
        }});

        renderer.domElement.addEventListener('mousedown', e => {{
            if (e.button === 0) isDragging = true;
            if (e.button === 2) isPanning = true;
            prevX = e.clientX; prevY = e.clientY;
        }});
        renderer.domElement.addEventListener('mousemove', e => {{
            const dx = e.clientX - prevX, dy = e.clientY - prevY;
            prevX = e.clientX; prevY = e.clientY;
            if (isDragging) {{
                euler.y -= dx * sensitivity;
                euler.x -= dy * sensitivity;
                euler.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, euler.x));
                camera.quaternion.setFromEuler(euler);
            }}
            if (isPanning) {{
                const right = new THREE.Vector3();
                const up = new THREE.Vector3();
                camera.getWorldDirection(new THREE.Vector3());
                right.setFromMatrixColumn(camera.matrixWorld, 0);
                up.setFromMatrixColumn(camera.matrixWorld, 1);
                camera.position.add(right.multiplyScalar(-dx * sensitivity * 0.5));
                camera.position.add(up.multiplyScalar(dy * sensitivity * 0.5));
            }}
        }});
        window.addEventListener('mouseup', () => {{ isDragging = false; isPanning = false; }});
        renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

        renderer.domElement.addEventListener('wheel', e => {{
            const dir = new THREE.Vector3();
            camera.getWorldDirection(dir);
            camera.position.add(dir.multiplyScalar(-e.deltaY * 0.003));
        }});

        document.addEventListener('keydown', e => {{
            keys[e.code] = true;
            if (e.code === 'KeyR') {{
                camera.position.set(...camPosT);
                camera.lookAt(...initT);
                euler.setFromQuaternion(camera.quaternion, 'YXZ');
            }}
        }});
        document.addEventListener('keyup', e => {{ keys[e.code] = false; }});

        let frames = 0, lastTime = performance.now();
        function animate() {{
            requestAnimationFrame(animate);

            const forward = new THREE.Vector3();
            camera.getWorldDirection(forward);
            const right = new THREE.Vector3();
            right.crossVectors(forward, camera.up).normalize();

            if (keys['KeyW']) camera.position.add(forward.clone().multiplyScalar(moveSpeed));
            if (keys['KeyS']) camera.position.add(forward.clone().multiplyScalar(-moveSpeed));
            if (keys['KeyA']) camera.position.add(right.clone().multiplyScalar(-moveSpeed));
            if (keys['KeyD']) camera.position.add(right.clone().multiplyScalar(moveSpeed));
            if (keys['KeyQ']) camera.position.y -= moveSpeed;
            if (keys['KeyE']) camera.position.y += moveSpeed;

            frames++;
            const now = performance.now();
            if (now - lastTime >= 1000) {{
                document.getElementById('fps-counter').textContent = frames + ' Hz';
                frames = 0; lastTime = now;
            }}

            const cp = camera.position;
            document.getElementById('cam-pose').textContent =
                cp.x.toFixed(2) + ' ' + cp.y.toFixed(2) + ' ' + cp.z.toFixed(2);

            renderer.render(scene, camera);
        }}
        animate();

        addEventListener('resize', () => {{
            camera.aspect = innerWidth / innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(innerWidth, innerHeight);
        }});
    </script>
</body>
</html>
        """

        return HTMLResponse(content=html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
