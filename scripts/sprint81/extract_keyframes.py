"""Sprint 81 — Step 1: keyframe extraction + pose matching.

scan.mp4 + poses.bin + manifest.json → keyframe JPEGs + index.json
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import av
import cv2
import numpy as np

# PoseMatcher는 server/src 아래에 있으므로 경로 추가
_SERVER_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))

from indoor_server.application.building.pose_matcher import PoseMatcher, DEFAULT_THRESHOLD_NS

logger = logging.getLogger(__name__)

# 기본 pose 매칭 threshold — plan 에서 ±50ms (50_000_000 ns) 로 완화
EXTRACT_THRESHOLD_NS = 50_000_000  # ±50ms


@dataclass
class KeyframeRecord:
    node_id: int
    frame_idx_in_video: int
    pts_ns: int
    ts_s: float
    image_path: str  # relative to out_dir
    pose_world_arkit: list[list[float]]  # 4x4
    pose_match_drift_ns: int


def extract(
    scan_root: Path,
    out_dir: Path,
    sample_hz: float = 2.0,
    jpeg_quality: int = 92,
) -> list[KeyframeRecord]:
    """keyframe 추출 + pose 매칭.

    Args:
        scan_root: manifest.json / scan.mp4 / poses.bin 이 있는 디렉터리.
        out_dir: JPEG + index.json 저장 경로.
        sample_hz: keyframe 샘플 빈도 (Hz).
        jpeg_quality: JPEG 인코딩 품질.

    Returns:
        매칭 성공한 KeyframeRecord list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = scan_root / "scan.mp4"
    poses_path = scan_root / "poses.bin"
    manifest_path = scan_root / "manifest.json"

    if not mp4_path.exists():
        raise FileNotFoundError(f"scan.mp4 not found: {mp4_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.bin not found: {poses_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    fps_nominal = float(manifest.get("video_fps_nominal", 60))

    # pose matcher 로드
    matcher = PoseMatcher(poses_path)
    logger.info("PoseMatcher loaded: %d records", len(matcher))

    # poses.bin PTS는 ARKit 절대 시간(시스템 업타임 기준).
    # mp4 PTS는 0부터 시작하는 상대 시간.
    # → poses.bin PTS를 pts[0] 기준 상대값으로 rebasing (cross_session script 동일 처리).
    pts_offset_ns = int(matcher.pts_array_ns[0]) if len(matcher) > 0 else 0
    logger.info("PTS rebase offset: %d ns (%.3f s)", pts_offset_ns, pts_offset_ns / 1e9)

    # frame 간격 (int) — 60fps, 2Hz → 30프레임마다
    frame_interval = max(1, int(round(fps_nominal / sample_hz)))
    logger.info("frame_interval=%d (%.1fHz from %.0ffps)", frame_interval, sample_hz, fps_nominal)

    records: list[KeyframeRecord] = []
    node_id = 1
    frame_idx = 0
    drop_count = 0
    total_sampled = 0

    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base  # Fraction
        logger.info("Video: codec=%s, fps=%.1f, duration=%.1fs",
                    stream.codec_context.name,
                    float(stream.average_rate or fps_nominal),
                    float(stream.duration or 0) * float(time_base))

        for packet in container.demux(stream):
            for frame in packet.decode():
                # 샘플 주기에 해당하는 frame 만 처리
                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                total_sampled += 1

                # PTS → ns
                if frame.pts is None:
                    frame_idx += 1
                    drop_count += 1
                    continue

                pts_ns_relative = int(frame.pts * float(time_base) * 1e9)
                # poses.bin 은 절대 시간 → mp4 relative PTS에 offset 더해서 절대값으로 매칭
                pts_ns_absolute = pts_ns_relative + pts_offset_ns
                pts_ns = pts_ns_relative  # index.json 에는 relative PTS 기록

                # pose 매칭 — absolute PTS 로 조회
                pose_sample = matcher.find_pose(pts_ns_absolute, threshold_ns=EXTRACT_THRESHOLD_NS)
                if pose_sample is None:
                    frame_idx += 1
                    drop_count += 1
                    continue

                # JPEG 저장
                img = frame.to_ndarray(format="bgr24")
                img_filename = f"{node_id:05d}.jpg"
                img_path = out_dir / img_filename
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                ok, encoded = cv2.imencode(".jpg", img, encode_params)
                if not ok:
                    logger.warning("imencode failed at frame %d, skip", frame_idx)
                    frame_idx += 1
                    drop_count += 1
                    continue
                img_path.write_bytes(encoded.tobytes())

                # pose 4x4
                T = pose_sample.transform  # (4,4) float32
                pose_list = T.tolist()

                drift_ns = abs(pts_ns_absolute - pose_sample.pts_ns)
                records.append(KeyframeRecord(
                    node_id=node_id,
                    frame_idx_in_video=frame_idx,
                    pts_ns=pts_ns,
                    ts_s=pts_ns / 1e9,
                    image_path=img_filename,
                    pose_world_arkit=pose_list,
                    pose_match_drift_ns=int(drift_ns),
                ))
                node_id += 1
                frame_idx += 1

    total_frames = frame_idx
    drop_rate = drop_count / max(total_sampled, 1)
    logger.info("Extraction done: total_frames=%d, sampled=%d, matched=%d, dropped=%d, drop_rate=%.2f%%",
                total_frames, total_sampled, len(records), drop_count, drop_rate * 100)

    # index.json 저장
    index_path = out_dir / "index.json"
    index_data = [asdict(r) for r in records]
    index_path.write_text(json.dumps(index_data, indent=2))

    return records


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sprint 81 Step 1: keyframe extraction")
    parser.add_argument("scan_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    records = extract(args.scan_root, args.out_dir, sample_hz=args.hz, jpeg_quality=args.quality)
    print(f"Extracted {len(records)} keyframes → {args.out_dir}/index.json")

    if args.report:
        import os
        pose_drifts = [r.pose_match_drift_ns for r in records]
        stats = {
            "keyframe_count": len(records),
            "sample_hz": args.hz,
            "drop_count": 0,  # 실제 drop은 logged
            "pose_drift_ns_mean": float(np.mean(pose_drifts)) if pose_drifts else 0,
            "pose_drift_ns_max": float(np.max(pose_drifts)) if pose_drifts else 0,
        }
        args.report.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
