/* IMDF Viewer — vanilla JS, SVG 기반 */
"use strict";

// ── 전역 상태 ──────────────────────────────────────────────────────────────────
let _token = "";
let _baseUrl = window.location.origin;

// 뷰포트 pan/zoom 상태
let _vb = { x: 0, y: 0, w: 100, h: 100 };  // viewBox 값
let _isDragging = false;
let _dragStart = { mx: 0, my: 0, vbx: 0, vby: 0 };

// 레이어 토글 상태
const _layers = {
  footprint: true,
  rectified: true,
  edges: false,
  nodes: false,
  amenity: true,
  anchor: true,
};

// ── DOM refs ────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── 초기화 ─────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  _initLayerCheckboxes();
  _initPanZoom();
  _fetchScanList();

  $("btn-load").addEventListener("click", onLoadClick);
  $("scan-select").addEventListener("change", onScanSelectChange);
});

function _initLayerCheckboxes() {
  Object.keys(_layers).forEach((key) => {
    const cb = $(`layer-${key}`);
    if (!cb) return;
    cb.checked = _layers[key];
    cb.addEventListener("change", () => {
      _layers[key] = cb.checked;
      const g = $(`g-${key}`);
      if (g) g.style.display = cb.checked ? "" : "none";
    });
  });
}

function _initPanZoom() {
  const vp = $("viewport");
  const svg = $("svg-root");

  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = vp.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // 마우스 위치를 중심으로 zoom
    const ratio = rect.width / _vb.w;
    const wx = _vb.x + mx / ratio;
    const wy = _vb.y + my / ratio;
    const factor = e.deltaY > 0 ? 1.15 : 0.87;
    _vb.w *= factor;
    _vb.h *= factor;
    _vb.x = wx - (mx / ratio) * factor;
    _vb.y = wy - (my / ratio) * factor;
    _applyViewBox(svg);
  }, { passive: false });

  vp.addEventListener("mousedown", (e) => {
    _isDragging = true;
    _dragStart = { mx: e.clientX, my: e.clientY, vbx: _vb.x, vby: _vb.y };
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!_isDragging) return;
    const rect = vp.getBoundingClientRect();
    const ratio = rect.width / _vb.w;
    _vb.x = _dragStart.vbx - (e.clientX - _dragStart.mx) / ratio;
    _vb.y = _dragStart.vby - (e.clientY - _dragStart.my) / ratio;
    _applyViewBox(svg);
  });

  window.addEventListener("mouseup", () => { _isDragging = false; });
}

function _applyViewBox(svg) {
  svg.setAttribute("viewBox", `${_vb.x} ${_vb.y} ${_vb.w} ${_vb.h}`);
}

// ── scan 목록 fetch ─────────────────────────────────────────────────────────────
async function _fetchScanList() {
  const tokenInput = $("token-input");
  _token = tokenInput ? tokenInput.value.trim() : "";

  try {
    const res = await fetch(`${_baseUrl}/dev/api/scans`, {
      headers: _token ? { Authorization: `Bearer ${_token}` } : {},
    });
    if (!res.ok) return;
    const scans = await res.json();
    const sel = $("scan-select");
    sel.innerHTML = '<option value="">— scan 선택 —</option>';
    scans.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.scan_id;
      opt.dataset.buildJobId = s.build_job_id;
      opt.dataset.walkableCells = s.walkable_cells;
      opt.dataset.mapNodes = s.map_nodes;
      opt.dataset.mapEdges = s.map_edges;
      const short = s.scan_id.slice(0, 8);
      opt.text = `${short}… (nodes=${s.map_nodes}, cells=${s.walkable_cells})`;
      sel.appendChild(opt);
    });
  } catch (_) {
    // 목록 fetch 실패는 조용히 무시 (토큰 없을 수 있음)
  }
}

function onScanSelectChange() {
  const sel = $("scan-select");
  const opt = sel.options[sel.selectedIndex];
  if (!opt || !opt.value) return;
  $("scan-id-input").value = opt.value;
  // HUD 미리 채우기
  _updateHud({
    scan_id: opt.value,
    build_job_id: opt.dataset.buildJobId || "—",
    walkable_cells: opt.dataset.walkableCells || "—",
    map_nodes: opt.dataset.mapNodes || "—",
    map_edges: opt.dataset.mapEdges || "—",
  });
}

// ── Load 버튼 ───────────────────────────────────────────────────────────────────
async function onLoadClick() {
  const scanId = $("scan-id-input").value.trim();
  const tokenInput = $("token-input");
  _token = tokenInput ? tokenInput.value.trim() : "";

  if (!scanId) { setStatus("scan_id를 입력하세요", "error"); return; }

  setStatus("로딩 중…", "");
  $("btn-load").disabled = true;

  try {
    const headers = _token ? { Authorization: `Bearer ${_token}` } : {};

    const [imdfRes, graphRes] = await Promise.all([
      fetch(`${_baseUrl}/scan/${scanId}/imdf`, { headers }),
      fetch(`${_baseUrl}/scan/${scanId}/graph`, { headers }),
    ]);

    if (!imdfRes.ok) {
      const err = await imdfRes.json().catch(() => ({}));
      throw new Error(`IMDF ${imdfRes.status}: ${err.code || imdfRes.statusText}`);
    }
    if (!graphRes.ok) {
      const err = await graphRes.json().catch(() => ({}));
      throw new Error(`Graph ${graphRes.status}: ${err.code || graphRes.statusText}`);
    }

    const [imdfBuf, graphData] = await Promise.all([
      imdfRes.arrayBuffer(),
      graphRes.json(),
    ]);

    const imdf = await _parseImdfZip(imdfBuf);
    await _render(scanId, imdf, graphData);
    setStatus("로드 완료", "ok");
  } catch (e) {
    setStatus(String(e.message || e), "error");
    console.error(e);
  } finally {
    $("btn-load").disabled = false;
  }
}

// ── IMDF zip 파싱 ───────────────────────────────────────────────────────────────
async function _parseImdfZip(buf) {
  const zip = await JSZip.loadAsync(buf);
  const result = {};
  for (const name of Object.keys(zip.files)) {
    const text = await zip.files[name].async("text");
    try {
      result[name] = JSON.parse(text);
    } catch (_) {
      result[name] = text;
    }
  }
  return result;
}

// ── 렌더링 ──────────────────────────────────────────────────────────────────────
async function _render(scanId, imdf, graphData) {
  const manifest = imdf["manifest.json"] || {};
  const footprint = imdf["footprint.geojson"] || { features: [] };
  const amenity = imdf["amenity.geojson"] || { features: [] };
  const anchor = imdf["anchor.geojson"] || { features: [] };
  const graphFeatures = (graphData.features || []);
  const props = graphData.properties || {};

  // HUD 갱신 (graph API는 geometry.type으로 구분: Point=node, LineString=edge)
  const nodeFeats = graphFeatures.filter((f) => f.geometry?.type === "Point");
  const edgeFeats = graphFeatures.filter((f) => f.geometry?.type === "LineString");
  _updateHud({
    scan_id: scanId,
    build_job_id: props.build_job_id || manifest.build_job_id || "—",
    walkable_cells: "—",
    map_nodes: nodeFeats.length,
    map_edges: edgeFeats.length,
    rectified: manifest.rectification?.accepted === true ? "yes" : "no",
  });

  // 모든 좌표 수집 → bounds 계산
  const allPts = _collectAllPoints(footprint, amenity, anchor, graphFeatures);
  if (allPts.length === 0) { setStatus("데이터 없음 — 좌표 추출 실패", "error"); return; }

  const bounds = _calcBounds(allPts);
  _fitViewBox(bounds);

  // SVG 그룹 초기화 후 렌더
  _clearGroups();

  _renderPolygons("g-footprint", footprint.features, "layer-footprint");
  if (manifest.rectification?.accepted === true) {
    _renderPolygons("g-rectified", footprint.features, "layer-rectified");
  }
  _renderEdges("g-edges", edgeFeats);
  _renderNodes("g-nodes", nodeFeats);
  _renderAmenity("g-amenity", amenity.features);
  _renderAnchor("g-anchor", anchor.features);

  // 레이어 가시성 동기화
  Object.keys(_layers).forEach((key) => {
    const g = $(`g-${key}`);
    if (g) g.style.display = _layers[key] ? "" : "none";
  });
}

// ── 좌표 수집 & bounds ───────────────────────────────────────────────────────────
function _collectAllPoints(...fcList) {
  const pts = [];
  const visit = (coords) => {
    if (typeof coords[0] === "number") {
      pts.push({ x: coords[0], y: coords[1] });
      return;
    }
    coords.forEach(visit);
  };
  fcList.forEach((fc) => {
    (fc.features || fc).forEach((f) => {
      if (f.geometry?.coordinates) visit(f.geometry.coordinates);
    });
  });
  return pts;
}

function _calcBounds(pts) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  pts.forEach(({ x, y }) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  });
  return { minX, minY, maxX, maxY };
}

function _fitViewBox(bounds) {
  const svg = $("svg-root");
  const vp = $("viewport");
  const vpW = vp.clientWidth || 800;
  const vpH = vp.clientHeight || 600;
  const dataW = bounds.maxX - bounds.minX || 1;
  const dataH = bounds.maxY - bounds.minY || 1;
  const margin = 0.08;
  const mw = dataW * margin;
  const mh = dataH * margin;

  // SVG Y축 반전: 데이터 y↑ → SVG y↓
  // SVG viewBox에서 Y를 뒤집어 표현: transform="scale(1,-1)" + translate
  // 간단한 방법: viewBox를 minY-(maxY+minY) 식으로 설정
  _vb = {
    x: bounds.minX - mw,
    y: -(bounds.maxY + mh),
    w: dataW + mw * 2,
    h: dataH + mh * 2,
  };

  // aspect ratio 보정
  const vpRatio = vpW / vpH;
  const dataRatio = _vb.w / _vb.h;
  if (dataRatio < vpRatio) {
    const extra = _vb.h * vpRatio - _vb.w;
    _vb.x -= extra / 2;
    _vb.w += extra;
  } else {
    const extra = _vb.w / vpRatio - _vb.h;
    _vb.y -= extra / 2;
    _vb.h += extra;
  }

  _applyViewBox(svg);
}

// ── SVG 그리기 헬퍼 ─────────────────────────────────────────────────────────────
const SVG_NS = "http://www.w3.org/2000/svg";

function _el(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}

// 좌표 [x, y, ?z] → SVG "x,y" (y 반전)
function _pt(coord) {
  return `${coord[0]},${-coord[1]}`;
}

// ring (좌표 배열) → SVG path d 문자열
function _ringToPath(ring) {
  return ring.map((c, i) => `${i === 0 ? "M" : "L"}${_pt(c)}`).join(" ") + " Z";
}

// MultiPolygon/Polygon → path d
function _geomToPath(geom) {
  if (!geom) return "";
  if (geom.type === "Polygon") {
    return geom.coordinates.map(_ringToPath).join(" ");
  }
  if (geom.type === "MultiPolygon") {
    return geom.coordinates
      .flatMap((poly) => poly.map(_ringToPath))
      .join(" ");
  }
  return "";
}

function _clearGroups() {
  ["footprint", "rectified", "edges", "nodes", "amenity", "anchor"].forEach((key) => {
    const g = $(`g-${key}`);
    if (g) g.innerHTML = "";
  });
}

function _renderPolygons(groupId, features, cssClass, colorByCategory = false) {
  const g = $(groupId);
  if (!g) return;
  features.forEach((feat, idx) => {
    const d = _geomToPath(feat.geometry);
    if (!d) return;
    const path = _el("path", { d, class: cssClass });
    if (colorByCategory) {
      const category = feat.properties?.category || "walkway";
      path.setAttribute("fill", _unitFill(category, idx));
      path.setAttribute("stroke", _unitStroke(category, idx));
    }
    g.appendChild(path);
  });
}

function _unitFill(category, idx) {
  const palette = {
    walkway: "rgba(248, 248, 244, 0.96)",
    block: "rgba(199, 195, 184, 0.66)",
    open_area: "rgba(88, 172, 116, 0.22)",
    pocket: "rgba(190, 128, 58, 0.24)",
    room: "rgba(232, 214, 143, 0.62)",
    lab: "rgba(180, 210, 245, 0.58)",
    office: "rgba(202, 188, 232, 0.58)",
    restroom: "rgba(120, 202, 220, 0.6)",
    stairs: "rgba(231, 159, 92, 0.68)",
    elevator: "rgba(168, 194, 94, 0.66)",
    entrance: "rgba(242, 180, 180, 0.62)",
    destination: "rgba(238, 205, 125, 0.62)",
  };
  return palette[category] || `hsla(${(idx * 67) % 360}, 55%, 48%, 0.22)`;
}

function _unitStroke(category, idx) {
  const palette = {
    walkway: "#c9c9c4",
    block: "#9f9a8e",
    open_area: "#58ac74",
    pocket: "#be803a",
    room: "#c7a43a",
    lab: "#5d93cc",
    office: "#8e78bd",
    restroom: "#3ca7ba",
    stairs: "#c46d24",
    elevator: "#7b9630",
    entrance: "#cf6565",
    destination: "#c6921f",
  };
  return palette[category] || `hsl(${(idx * 67) % 360}, 55%, 55%)`;
}

function _renderEdges(groupId, features) {
  const g = $(groupId);
  if (!g) return;
  features.forEach((feat) => {
    const coords = feat.geometry?.coordinates;
    if (!coords || coords.length < 2) return;
    const pts = _manhattanizeCoords(coords, 0.25).map(_pt).join(" ");
    const polyline = _el("polyline", {
      points: pts,
      class: "layer-edges",
      fill: "none",
      stroke: "#27ae60",
      "stroke-width": "0.5",
    });
    g.appendChild(polyline);
  });
}

function _manhattanizeCoords(coords, grid = 0.25) {
  if (!Array.isArray(coords) || coords.length <= 1) return coords || [];
  const snapped = coords.map((coord) => _snapCoord(coord, grid));
  const out = [snapped[0]];
  for (let i = 1; i < snapped.length; i += 1) {
    const prev = out[out.length - 1];
    const next = snapped[i];
    const dx = Math.abs(next[0] - prev[0]);
    const dy = Math.abs(next[1] - prev[1]);
    if (dx > 1e-9 && dy > 1e-9) {
      const bend = dx >= dy
        ? [next[0], prev[1], next[2] ?? prev[2] ?? 0]
        : [prev[0], next[1], next[2] ?? prev[2] ?? 0];
      _pushUniqueCoord(out, bend);
    }
    _pushUniqueCoord(out, next);
  }
  return out;
}

function _snapCoord(coord, grid) {
  return [
    Math.round(coord[0] / grid) * grid,
    Math.round(coord[1] / grid) * grid,
    coord[2] ?? 0,
  ];
}

function _pushUniqueCoord(out, coord) {
  const prev = out[out.length - 1];
  if (prev && Math.abs(prev[0] - coord[0]) < 1e-9 && Math.abs(prev[1] - coord[1]) < 1e-9) {
    return;
  }
  out.push(coord);
}

function _nodeClass(nodeType) {
  if (!nodeType) return "node-skeleton";
  if (nodeType === "poi") return "node-poi";
  if (nodeType === "poi_attach_virtual") return "node-poi-virtual";
  if (nodeType === "junction") return "node-junction";
  if (nodeType === "endpoint") return "node-endpoint";
  return "node-skeleton";
}

function _nodeRadius(nodeType) {
  if (nodeType === "poi") return 1.5;
  if (nodeType === "poi_attach_virtual") return 1.0;
  return 0.8;
}

function _renderNodes(groupId, features) {
  const g = $(groupId);
  if (!g) return;
  features.forEach((feat) => {
    const coords = feat.geometry?.coordinates;
    if (!coords) return;
    const props = feat.properties || {};
    const nodeType = props.node_type || "";
    const r = _nodeRadius(nodeType);
    const cx = coords[0];
    const cy = -coords[1];
    const circle = _el("circle", {
      cx,
      cy,
      r,
      class: _nodeClass(nodeType),
    });
    // tooltip on hover
    const tipLines = [
      `node_id: ${(props.node_id || "").slice(0, 8)}…`,
      `type: ${nodeType}`,
      props.label ? `label: ${props.label}` : "",
      `x=${Number(coords[0]).toFixed(2)} y=${Number(coords[1]).toFixed(2)} z=${Number(coords[2] || 0).toFixed(2)}`,
    ].filter(Boolean).join("\n");
    circle.addEventListener("mouseenter", (e) => showTooltip(e, tipLines));
    circle.addEventListener("mouseleave", hideTooltip);
    circle.addEventListener("mousemove", moveTooltip);
    g.appendChild(circle);
  });
}

function _renderAmenity(groupId, features) {
  const g = $(groupId);
  if (!g) return;
  features.forEach((feat) => {
    const props = feat.properties || {};
    const coords = Array.isArray(props.display_point) ? props.display_point : feat.geometry?.coordinates;
    if (!coords) return;
    const name = props.name?.ko || props.name || "";
    const category = props.category || "unknown";
    const cx = coords[0];
    const cy = -coords[1];
    const color = _amenityColor(category);
    const glyph = _amenityGlyph(category);
    const circle = _el("circle", {
      cx,
      cy,
      r: 1.9,
      fill: color,
      stroke: "#111",
      "stroke-width": "0.25",
    });
    const tipLines = [
      `${glyph} ${name || category}`,
      `category: ${category}`,
      props.connector_key ? `connector: ${props.connector_key}` : "",
      props.route_node_id ? `route_node: ${String(props.route_node_id).slice(0, 8)}…` : "",
      props.display_area_id ? `area: ${props.display_area_id}` : "",
      props.display_point_role ? `display: ${props.display_point_role}` : "",
      props.analysis?.confidence != null ? `confidence: ${Number(props.analysis.confidence).toFixed(2)}` : "",
    ].filter(Boolean).join("\n");
    circle.addEventListener("mouseenter", (e) => showTooltip(e, tipLines));
    circle.addEventListener("mouseleave", hideTooltip);
    circle.addEventListener("mousemove", moveTooltip);
    g.appendChild(circle);
    const icon = _el("text", {
      x: cx,
      y: cy + 0.65,
      "text-anchor": "middle",
      "font-size": "1.8",
      fill: "#111",
    });
    icon.textContent = glyph;
    g.appendChild(icon);
    // 라벨
    if (name) {
      const text = _el("text", { x: cx + 2, y: cy, class: "layer-amenity" });
      text.textContent = name;
      g.appendChild(text);
    }
  });
}

function _amenityColor(category) {
  const palette = {
    room: "#d6a526",
    lab: "#4a90d9",
    office: "#9b7bd2",
    restroom: "#31a7bd",
    stairs: "#e67e22",
    elevator: "#93b33d",
    entrance: "#e06b6b",
    destination: "#f1c40f",
  };
  return palette[category] || "#e74c3c";
}

function _amenityGlyph(category) {
  const glyphs = {
    room: "R",
    lab: "L",
    office: "O",
    restroom: "W",
    stairs: "S",
    elevator: "E",
    entrance: "D",
    destination: "P",
  };
  return glyphs[category] || "P";
}

function _renderAnchor(groupId, features) {
  const g = $(groupId);
  if (!g) return;
  features.forEach((feat) => {
    const coords = feat.geometry?.coordinates;
    if (!coords) return;
    const cx = coords[0];
    const cy = -coords[1];
    const sz = 2;
    // 십자 표시
    g.appendChild(_el("line", { x1: cx - sz, y1: cy, x2: cx + sz, y2: cy, stroke: "#f1c40f", "stroke-width": "0.4" }));
    g.appendChild(_el("line", { x1: cx, y1: cy - sz, x2: cx, y2: cy + sz, stroke: "#f1c40f", "stroke-width": "0.4" }));
    // 좌표 라벨
    const text = _el("text", { x: cx + sz + 0.3, y: cy, fill: "#f1c40f", "font-size": "1.8" });
    text.textContent = `(${Number(coords[0]).toFixed(1)}, ${Number(coords[1]).toFixed(1)})`;
    g.appendChild(text);
  });
}

// ── HUD 업데이트 ────────────────────────────────────────────────────────────────
function _updateHud(info) {
  const set = (id, val) => { const el = $(id); if (el) el.textContent = val; };
  set("hud-scan-id", typeof info.scan_id === "string" ? info.scan_id.slice(0, 8) + "…" : "—");
  set("hud-build-job", typeof info.build_job_id === "string" ? info.build_job_id.slice(0, 8) + "…" : "—");
  set("hud-walkable", info.walkable_cells ?? "—");
  set("hud-nodes", info.map_nodes ?? "—");
  set("hud-edges", info.map_edges ?? "—");
  set("hud-rectified", info.rectified ?? "—");
}

// ── tooltip ────────────────────────────────────────────────────────────────────
function showTooltip(e, text) {
  const tip = $("tooltip");
  tip.textContent = text;
  tip.style.display = "block";
  moveTooltip(e);
}
function hideTooltip() { $("tooltip").style.display = "none"; }
function moveTooltip(e) {
  const tip = $("tooltip");
  const vp = $("viewport");
  const rect = vp.getBoundingClientRect();
  let lx = e.clientX - rect.left + 12;
  let ly = e.clientY - rect.top + 12;
  if (lx + 200 > rect.width) lx -= 220;
  tip.style.left = lx + "px";
  tip.style.top = ly + "px";
}

// ── 상태 메시지 ─────────────────────────────────────────────────────────────────
function setStatus(msg, cls) {
  const el = $("status-msg");
  el.textContent = msg;
  el.className = cls || "";
}
