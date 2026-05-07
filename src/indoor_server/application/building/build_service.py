"""BuildService — 빌드 유스케이스 진입점."""
from __future__ import annotations

import asyncio
import logging
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from indoor_server.application.api_v1.poi_catalog_service import POICatalogService
from indoor_server.application.api_v1.vertical_connector_catalog_service import (
    VerticalConnectorCatalogService,
)
from indoor_server.application.building.keyframe_node_id_backfill import (
    backfill_keyframe_node_ids,
)
from indoor_server.application.building.pipeline import BuildPipeline
from indoor_server.application.building.pose_backfill import (
    PoseBackfillStats,
    run_full_backfill,
)
from indoor_server.application.building.reprocess_service import (
    RTABMapReprocessError,
    RTABMapReprocessRunner,
)
from indoor_server.application.building.steps.floor_segmentation import KeyframeRef
from indoor_server.application.rtabmap.reader import RtabmapReader
from indoor_server.domain.building.enums import (
    BuildFailureReason,
    BuildState,
    BuildStep,
    EdgeType,
    NodeType,
)
from indoor_server.domain.building.models import BuildCounts, MapEdgeVO, MapNodeVO
from indoor_server.domain.building.rtabmap_models import (
    RtabmapDataFrame,
    RtabmapDiagnostics,
    RtabmapFeaturePoint,
    RtabmapLink,
    RtabmapNode,
)
from indoor_server.domain.scan.models import POIMarkRow
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.interfloor_mark_repo import (
    InterfloorMarkDbRow,
    InterfloorMarkRepository,
)
from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository
from indoor_server.infrastructure.db.repositories.poi_mark_repo import POIMarkRepository

logger = logging.getLogger(__name__)


class BuildService:
    def __init__(
        self,
        engine: AsyncEngine,
        pipeline: BuildPipeline,
        debug_root: Path | None = None,
        reprocess_runner: RTABMapReprocessRunner | None = None,
    ) -> None:
        self._engine = engine
        self._pipeline = pipeline
        self._debug_root = debug_root
        # Sprint 65 Phase 4c: rtabmap-reprocess subprocess runner.
        # 미주입 시 default 인스턴스 생성 (binary 가 없으면 reprocess phase 자동 SKIP).
        self._reprocess_runner = reprocess_runner or RTABMapReprocessRunner()

    async def run(self, *, scan_id: str, build_job_id: str) -> None:
        """worker가 claim 직후 호출. 예외는 failed 상태로 기록 후 삼킴."""
        try:
            await self._do_run(scan_id=scan_id, build_job_id=build_job_id)
        except Exception:
            tb = traceback.format_exc()
            logger.error(
                "build failed scan_id=%s build_job_id=%s\n%s",
                scan_id, build_job_id, tb
            )
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    repo = BuildJobRepository(session)
                    await repo.update_state(
                        build_job_id=build_job_id,
                        state=BuildState.FAILED,
                        failure_reason=BuildFailureReason.INTERNAL,
                        failure_detail=tb[:2000],
                        finished_at=datetime.now(tz=UTC),
                        step=BuildStep.DONE,
                        progress=1.0,
                    )

    async def _do_run(self, *, scan_id: str, build_job_id: str) -> None:
        """실제 파이프라인 실행."""
        from indoor_server.application.building.debug.filesystem_sink import FilesystemDebugSink
        from indoor_server.config import settings

        rtabmap_db_path = settings.storage_root / "scans" / scan_id / "rtabmap.db"

        # Sprint 49 (Codex BLOCKER 5): manifest.json 이 storage 에 있으면 재파싱.
        manifest_metadata: dict[str, object] | None = None
        manifest_path = (
            settings.storage_root / "scans" / scan_id / "manifest.json"
        )
        manifest_mode: str | None = None
        manifest_obj = None
        if manifest_path.exists():
            try:
                from indoor_server.application.scan_manifest import parse_manifest_file

                manifest_obj = parse_manifest_file(
                    manifest_path, expected_scan_id=scan_id
                )
                manifest_metadata = manifest_obj.to_metadata()
                manifest_mode = manifest_obj.mode
            except Exception as e:
                logger.warning(
                    "manifest reparse failed scan_id=%s err=%s", scan_id, e
                )
                manifest_metadata = None
                manifest_obj = None

        # Sprint 67/74: raw_video_recording (v7) 모드면 dense video floor evidence 를 부가로 dump.
        # 메인 build_pipeline 은 기존 rtabmap.db 기반 그대로 유지한다.
        # evidence 는 var/debug/{scan_id}/{build_job_id}/dense_video_floor/ 에 저장된다.
        # 향후 sprint 에서 dense floor → polygon source 로 승격 가능.
        if manifest_obj is not None and manifest_obj.is_video_mode:
            try:
                await self._run_dense_video_floor_evidence(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                    manifest=manifest_obj,
                )
            except Exception as e:
                logger.warning(
                    "dense_video_floor evidence dump failed scan_id=%s err=%s — build 계속 진행",
                    scan_id, e,
                )

        # Sprint 65 Issue 2 fix: raw recording 모드에서는 iOS finalize backfill 이 빈 결과
        # (Mem/STMSize=1 + Mem/RehearsalSimilarity=1.0) 이므로 keyframe_meta.rtabmap_node_id 가
        # NULL 인 채로 ingest 된다. 서버에서 rtabmap.db Node.stamp 매칭으로 직접 backfill.
        # idempotent — NULL 행만 갱신, 이미 set 된 행은 건드리지 않음.
        node_id_stats = None
        try:
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    node_id_stats = await backfill_keyframe_node_ids(
                        session,
                        scan_id=scan_id,
                        rtabmap_db_path=rtabmap_db_path,
                    )
        except Exception as e:
            logger.warning(
                "keyframe nodeID backfill 실패 scan_id=%s err=%s — build 계속 진행",
                scan_id, e,
            )

        # Sprint 65 Phase 4c: Raw ARKit recording 모드 scan 은 desktop rtabmap-reprocess 로
        # SIFT 재추출 + loop closure + graph optimization 후 keyframe/poi/interfloor pose 보정.
        # 이후 build_pipeline 은 reprocessed db + 보정된 pose 로 동작 → ARKit drift 영향 제거.
        reprocess_meta = await self._maybe_reprocess(
            scan_id=scan_id,
            input_db=rtabmap_db_path,
            manifest_mode=manifest_mode,
        )
        if reprocess_meta is not None and reprocess_meta.get("output_db_path"):
            rtabmap_db_path = Path(str(reprocess_meta["output_db_path"]))

        async with AsyncSession(self._engine) as session:
            keyframes = await self._load_keyframes(session, scan_id)
            pois = await self._load_pois(session, scan_id)
            interfloor_marks = await InterfloorMarkRepository(session).list_by_scan(
                scan_id
            )
        rtabmap_diag = await asyncio.to_thread(
            self._inspect_rtabmap,
            rtabmap_db_path,
            keyframes,
        )
        # Sprint 77 cycle 2: video-mode 분기 (raw_video_recording + rtabmap not ready)
        # manifest가 video mode이고 rtabmap DB가 ready 아니면 sprint82 pipeline으로 라우팅.
        if (
            manifest_obj is not None
            and manifest_obj.is_video_mode
            and not rtabmap_diag.ready
        ):
            await self._run_video_mode_build(
                scan_id=scan_id,
                build_job_id=build_job_id,
                manifest_obj=manifest_obj,
                interfloor_marks=interfloor_marks,
            )
            return

        if settings.rtabmap_build_required and not rtabmap_diag.ready:
            await self._fail_rtabmap_not_ready(
                scan_id=scan_id,
                build_job_id=build_job_id,
                diagnostics=rtabmap_diag,
            )
            return

        rtabmap_nodes: list[RtabmapNode] = []
        rtabmap_links: list[RtabmapLink] = []
        rtabmap_features: list[RtabmapFeaturePoint] = []
        rtabmap_frames: list[RtabmapDataFrame] = []
        if rtabmap_diag.ready:
            (
                rtabmap_nodes,
                rtabmap_links,
                rtabmap_features,
                rtabmap_frames,
            ) = await asyncio.to_thread(
                self._load_rtabmap_evidence,
                rtabmap_db_path,
            )

        scan_uuid = UUID(scan_id)
        job_uuid = UUID(build_job_id)

        async def progress_sink(step: BuildStep, p: float) -> None:
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    await BuildJobRepository(session).update_progress(
                        build_job_id=build_job_id,
                        step=step,
                        progress=p,
                    )

        async def cancel_check() -> bool:
            async with AsyncSession(self._engine) as session:
                state = await BuildJobRepository(session).get_current_state(build_job_id)
            return bool(state == BuildState.CANCELLED)

        debug_sink: FilesystemDebugSink | None = None
        if self._debug_root is not None:
            out_dir = self._debug_root / scan_id / build_job_id
            debug_sink = FilesystemDebugSink(
                out_dir=out_dir,
                storage_root=settings.storage_root,
            )

        try:
            outcome = await self._pipeline.execute(
                scan_id=scan_uuid,
                build_job_id=job_uuid,
                keyframes=keyframes,
                pois=pois,
                rtabmap_nodes=rtabmap_nodes,
                rtabmap_links=rtabmap_links,
                rtabmap_features=rtabmap_features,
                rtabmap_frames=rtabmap_frames,
                scan_manifest_metadata=manifest_metadata,
                progress_sink=progress_sink,
                cancel_check=cancel_check,
                debug_sink=debug_sink,
            )
        finally:
            if debug_sink is not None:
                try:
                    debug_sink.finalize()
                except Exception as e:
                    logger.warning("debug sink finalize failed: %s", e)

        # Sprint 65 Phase 4c: counts 에 reprocess 메타 노출 (downstream evidence).
        rtabmap_counts = rtabmap_diag.to_counts_dict()
        if reprocess_meta is not None:
            # path 객체는 JSON 직렬화 위해 string 으로.
            sanitized_meta = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in reprocess_meta.items()
            }
            rtabmap_counts = {**rtabmap_counts, "reprocess": sanitized_meta}
        # Sprint 65 Issue 2 fix: nodeID backfill 통계 노출.
        if node_id_stats is not None:
            rtabmap_counts = {
                **rtabmap_counts,
                "node_id_backfill": {
                    "matched": node_id_stats.matched,
                    "skipped_no_match": node_id_stats.skipped_no_match,
                    "already_set": node_id_stats.already_set,
                },
            }

        outcome = outcome.model_copy(
            update={
                "counts": outcome.counts.model_copy(
                    update={
                        "build_source": (
                            outcome.counts.build_source
                            or (
                                "legacy_geometry_rtabmap_validated"
                                if rtabmap_diag.ready
                                else "legacy_geometry_rtabmap_unvalidated"
                            )
                        ),
                        "rtabmap": rtabmap_counts,
                    }
                )
            }
        )

        if outcome.passed_quality_gate and interfloor_marks:
            nodes_with_connectors, edges_with_connectors = _append_interfloor_connector_nodes(
                interfloor_marks=interfloor_marks,
                nodes=outcome.nodes,
                edges=outcome.edges,
                scan_id=scan_uuid,
                build_job_id=job_uuid,
            )
            outcome = outcome.model_copy(
                update={
                    "nodes": nodes_with_connectors,
                    "edges": edges_with_connectors,
                    "counts": outcome.counts.model_copy(
                        update={
                            "map_nodes": len(nodes_with_connectors),
                            "map_edges": len(edges_with_connectors),
                        }
                    ),
                }
            )

        if not outcome.passed_quality_gate:
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    await BuildJobRepository(session).update_state(
                        build_job_id=build_job_id,
                        state=BuildState.FAILED,
                        step=BuildStep.QUALITY_GATE,
                        progress=0.95,
                        failure_reason=outcome.failure_reason,
                        counts=outcome.counts,
                        finished_at=datetime.now(tz=UTC),
                    )
            return

        # PERSIST
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                await BuildJobRepository(session).update_progress(
                    build_job_id=build_job_id,
                    step=BuildStep.PERSIST,
                    progress=0.97,
                )
                await MapGraphRepository(session).replace_graph(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                    nodes=outcome.nodes,
                    edges=outcome.edges,
                )
                # POI world_pose 복사
                updates = [
                    (poi_id, x, y, z)
                    for poi_id, (x, y, z) in outcome.poi_world_poses.items()
                ]
                await POIMarkRepository(session).set_world_pose(updates=updates)
                await POICatalogService(session).sync_scan_pois(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                )
                await VerticalConnectorCatalogService(session).sync_scan_interfloor_marks(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                )

                await BuildJobRepository(session).update_state(
                    build_job_id=build_job_id,
                    state=BuildState.SUCCEEDED,
                    step=BuildStep.DONE,
                    progress=1.0,
                    counts=outcome.counts,
                    finished_at=datetime.now(tz=UTC),
                )

        logger.info(
            "build succeeded scan_id=%s nodes=%d edges=%d pois=%d",
            scan_id,
            len(outcome.nodes),
            len(outcome.edges),
            len(outcome.poi_world_poses),
        )

    async def _load_keyframes(
        self, session: AsyncSession, scan_id: str
    ) -> list[KeyframeRef]:
        rows = (
            await session.execute(
                sa.select(t.keyframe_meta).where(t.keyframe_meta.c.scan_id == scan_id)
            )
        ).fetchall()

        scan_uuid = UUID(scan_id)
        # image_path는 사이드카 기준 상대 경로(예: "keyframes/000001.jpg").
        # pipeline은 storage_root 기준이므로 scan storage prefix를 붙여 절대화한다.
        scan_prefix = f"scans/{scan_id}"
        return [
            KeyframeRef(
                scan_id=scan_uuid,
                seq=row.seq,
                image_path=f"{scan_prefix}/{row.image_path}",
                tx=row.tx,
                ty=row.ty,
                tz=row.tz,
                pose_matrix=row.pose_matrix,
                rtabmap_node_id=row.rtabmap_node_id,
            )
            for row in rows
        ]

    async def _load_pois(
        self, session: AsyncSession, scan_id: str
    ) -> list[POIMarkRow]:
        return await POIMarkRepository(session).list_by_scan(scan_id)

    def _inspect_rtabmap(
        self,
        db_path: Path,
        keyframes: list[KeyframeRef],
    ) -> RtabmapDiagnostics:
        diag = RtabmapReader().inspect(
            db_path,
            keyframe_node_ids=[kf.rtabmap_node_id for kf in keyframes],
        )
        logger.info(
            "rtabmap inspect scan_db=%s ready=%s nodes=%d data=%d features=%d "
            "kf_coverage=%.3f issues=%s",
            db_path,
            diag.ready,
            diag.node_count,
            diag.data_count,
            diag.feature_count,
            diag.keyframe_node_coverage,
            diag.issues,
        )
        return diag

    def _load_rtabmap_evidence(
        self,
        db_path: Path,
    ) -> tuple[
        list[RtabmapNode],
        list[RtabmapLink],
        list[RtabmapFeaturePoint],
        list[RtabmapDataFrame],
    ]:
        reader = RtabmapReader()
        nodes = reader.load_nodes(db_path)
        links = reader.load_links(db_path)
        features = reader.load_feature_points(db_path)
        frames = reader.load_data_frames(db_path)
        logger.info(
            "rtabmap evidence loaded db=%s nodes=%d links=%d features=%d frames=%d",
            db_path,
            len(nodes),
            len(links),
            len(features),
            len(frames),
        )
        return nodes, links, features, frames

    async def _maybe_reprocess(
        self,
        *,
        scan_id: str,
        input_db: Path,
        manifest_mode: str | None,
    ) -> dict[str, object] | None:
        """Sprint 65 Phase 4c: raw ARKit recording 모드 scan 에 대한 desktop reprocess.

        조건:
            - manifest.mode == "raw_arkit_recording"
            - rtabmap-reprocess binary 가용
            - reprocessed.db 가 아직 없음 (idempotent)

        실패 시 graceful degradation: log + raw db 그대로 build_pipeline 진입.
        """
        if manifest_mode != "raw_arkit_recording":
            return None
        if not self._reprocess_runner.is_available():
            logger.warning(
                "rtabmap-reprocess binary 미가용 — raw scan 을 그대로 빌드 진행 scan_id=%s",
                scan_id,
            )
            return {
                "status": "skipped",
                "reason": "binary_not_available",
            }
        if not input_db.exists():
            return {"status": "skipped", "reason": "input_db_missing"}

        output_db = input_db.parent / "rtabmap_reprocessed.db"
        # 이미 처리된 scan 은 skip (idempotent). force-rebuild 시 호출자가 미리 삭제.
        if output_db.exists():
            return {
                "status": "skipped",
                "reason": "already_reprocessed",
                "output_db_path": output_db,
            }

        try:
            result = await self._reprocess_runner.run(
                input_db=input_db, output_db=output_db
            )
        except RTABMapReprocessError as e:
            logger.warning(
                "rtabmap-reprocess 실패 — raw db 로 fallback scan_id=%s err=%s",
                scan_id, e,
            )
            return {"status": "failed", "reason": str(e)[:500]}

        backfill_stats: PoseBackfillStats | None = None
        try:
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    backfill_stats = await run_full_backfill(
                        session,
                        scan_id=scan_id,
                        optimized=result.optimized_poses,
                    )
        except Exception as e:
            logger.warning(
                "pose backfill 실패 (rtabmap-reprocess 자체는 성공). scan_id=%s err=%s",
                scan_id, e,
            )

        return {
            "status": "succeeded",
            "output_db_path": output_db,
            "duration_s": round(result.duration_s, 3),
            "optimized_node_count": len(result.optimized_poses),
            "loop_closures": result.total_loop_closures,
            "backfill_stats": (
                {
                    "keyframe_updated": backfill_stats.keyframe_updated,
                    "keyframe_skipped_no_node_id": backfill_stats.keyframe_skipped_no_node_id,
                    "poi_updated": backfill_stats.poi_updated,
                    "branch_updated": backfill_stats.branch_updated,
                    "interfloor_updated": backfill_stats.interfloor_updated,
                }
                if backfill_stats is not None
                else None
            ),
        }

    async def _run_dense_video_floor_evidence(
        self,
        *,
        scan_id: str,
        build_job_id: str,
        manifest: object,  # ScanManifest (forward import 회피용 object 타입)
    ) -> None:
        """Sprint 67 Phase 7 — raw_video_recording 모드 부가 evidence dump.

        scan.mp4 + poses.bin + manifest intrinsics → DenseVideoFloorStep → world point cloud.
        결과는 var/debug/{scan_id}/{build_job_id}/dense_video_floor/ 에 NPZ + JSON 으로 저장.
        실패해도 main pipeline 은 계속 (rtabmap.db 기반 기존 path 가 살아 있음).
        """
        from indoor_server.application.building.steps.back_projection import Intrinsics
        from indoor_server.application.building.steps.dense_video_floor import (
            DenseVideoFloorParams,
            DenseVideoFloorStep,
        )
        from indoor_server.config import settings

        scan_root = settings.storage_root / "scans" / scan_id
        video_path = scan_root / "scan.mp4"
        poses_path = scan_root / "poses.bin"
        if not video_path.exists() or not poses_path.exists():
            logger.warning(
                "dense_video_floor: video/poses 파일 부재 — skip scan_id=%s "
                "video=%s poses=%s",
                scan_id, video_path.exists(), poses_path.exists(),
            )
            return

        # manifest 에서 intrinsics 가 채워져 있어야 함 (iOS makeV7 가 무조건 채움).
        fx = getattr(manifest, "intrinsics_fx", None)
        fy = getattr(manifest, "intrinsics_fy", None)
        cx = getattr(manifest, "intrinsics_cx", None)
        cy = getattr(manifest, "intrinsics_cy", None)
        if fx is None or fy is None or cx is None or cy is None:
            logger.warning(
                "dense_video_floor: manifest intrinsics 누락 — skip scan_id=%s",
                scan_id,
            )
            return

        intrinsics = Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

        # segformer 모델 로드. ModelCache 로 hf 캐시 또는 다운로드.
        # main pipeline 의 segmenter 와 별도 인스턴스 (onnx 는 메모리 적게 쓴다).
        from indoor_server.infrastructure.ml.model_cache import ModelCache
        from indoor_server.infrastructure.ml.segformer_onnx import (
            SegformerOnnxSegmenter,
        )

        try:
            model_cache = ModelCache(
                settings.model_cache_dir,
                repo_id=settings.segformer_model_repo_id,
                filename=settings.segformer_model_filename,
            )
            model_path = await asyncio.to_thread(model_cache.ensure)
        except Exception as e:
            logger.warning(
                "dense_video_floor: segformer model 로드 실패 — skip err=%s", e,
            )
            return
        segmenter = SegformerOnnxSegmenter(model_path=model_path)
        step = DenseVideoFloorStep(
            segmenter=segmenter,
            params=DenseVideoFloorParams(stride=2, pixel_stride=4),
        )
        # z0 추정은 첫 frame pose 의 height - 1.5m (camera 높이 가정).
        # 정확한 z0 는 main pipeline 의 RTAB-Map trajectory 에서 더 잘 산출됨.
        from indoor_server.application.building.pose_matcher import PoseMatcher

        pm = PoseMatcher(poses_path)
        first_pose = pm.all_samples()[0] if len(pm) > 0 else None
        if first_pose is None:
            logger.warning("dense_video_floor: poses.bin 비어있음 — skip")
            return
        z0_estimate = float(first_pose.translation[2]) - 1.5

        cloud = await step.run(
            video_path=video_path,
            poses_path=poses_path,
            intrinsics=intrinsics,
            z0=z0_estimate,
        )

        # evidence dump
        evidence_dir = (
            (self._debug_root or settings.storage_root / "debug")
            / scan_id / build_job_id / "dense_video_floor"
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)

        import json

        import numpy as np

        np.savez_compressed(
            evidence_dir / "world_points.npz",
            points_xy=cloud.points_xy.astype(np.float32),
            z_values=cloud.z_values.astype(np.float32),
        )
        (evidence_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "z0_estimate_m": z0_estimate,
                    "world_point_count": int(cloud.points_xy.shape[0]),
                    "intrinsics": {
                        "fx": intrinsics.fx, "fy": intrinsics.fy,
                        "cx": intrinsics.cx, "cy": intrinsics.cy,
                    },
                    **cloud.metadata,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "dense_video_floor evidence saved scan_id=%s build_job=%s "
            "world_points=%d → %s",
            scan_id, build_job_id, int(cloud.points_xy.shape[0]),
            evidence_dir,
        )

    # ── Sprint 77 cycle 2: video-mode build ──────────────────────────────────

    async def _run_video_mode_build(
        self,
        *,
        scan_id: str,
        build_job_id: str,
        manifest_obj: object,
        interfloor_marks: list[InterfloorMarkDbRow],
    ) -> None:
        """raw_video_recording 모드 전용 build 경로.

        sprint82 pipeline (polygon + nav graph) 을 실행하여 결과를 DB 에 저장한다.
        rtabmap quality gate 를 우회하고, nodes≥3 / edges≥nodes-1 의 별도 gate 를 사용한다.
        """
        from indoor_server.application.building.video_mode_pipeline import run_sprint82
        from indoor_server.config import settings

        logger.info(
            "video_mode_pipeline: START scan_id=%s build_job_id=%s",
            scan_id, build_job_id,
        )

        # progress update: 시작
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                await BuildJobRepository(session).update_progress(
                    build_job_id=build_job_id,
                    step=BuildStep.FLOOR_SEG,
                    progress=0.1,
                )

        # merged DB 경로: 단일 scan 이면 zip 내 rtabmap.db 를 그대로 사용
        merged_db = settings.storage_root / "scans" / scan_id / "rtabmap.db"

        # 빈 rtabmap.db (ARKit 클라가 raw 만 보낸 경우) 감지 시 mp4 + poses.bin 으로 seed
        from indoor_server.application.building.rtabmap_seeder import (
            is_empty_rtabmap_db,
            seed_rtabmap_db_from_video,
        )
        if is_empty_rtabmap_db(merged_db):
            scan_root = settings.storage_root / "scans" / scan_id
            logger.info(
                "rtabmap.db is empty — seeding from mp4+poses scan_id=%s", scan_id,
            )
            try:
                node_count = await asyncio.to_thread(
                    seed_rtabmap_db_from_video,
                    scan_root=scan_root,
                    output_db=merged_db,
                )
                logger.info(
                    "rtabmap_seeder: success scan_id=%s nodes=%d", scan_id, node_count,
                )
                # seed 후 reprocess 로 Word/Feature/GlobalDesc 채움
                from indoor_server.application.api_v1.scan_compat_service import (
                    _run_rtabmap_reprocess_single,
                )
                seeded_db = merged_db
                reprocessed_db = merged_db.with_suffix(".reprocessed.db")
                await _run_rtabmap_reprocess_single(
                    binary_path="rtabmap-reprocess",
                    input_db=seeded_db,
                    output_db=reprocessed_db,
                    timeout_s=600.0,
                )
                # 원본 자리에 reprocess 결과 덮어쓰기 (rtab-map graph optimization
                # 결과 pose 를 그대로 사용 — frame 일관성 보장)
                seeded_db.unlink()
                reprocessed_db.rename(seeded_db)
                logger.info(
                    "rtabmap-reprocess: done scan_id=%s db=%s", scan_id, seeded_db,
                )
                # Seed + reprocess 가 Node 를 채웠으므로 keyframe_meta.rtabmap_node_id
                # 를 위치 기반 매칭으로 backfill (Node.stamp 가 mp4 relative 라 기존
                # stamp-based backfill 은 매칭 안 됨).
                from indoor_server.application.building.rtabmap_seeder import (
                    backfill_keyframe_node_ids_by_position,
                )
                try:
                    async with AsyncSession(self._engine) as session:
                        async with session.begin():
                            stats = await backfill_keyframe_node_ids_by_position(
                                session=session,
                                scan_id=scan_id,
                                rtabmap_db_path=seeded_db,
                            )
                    logger.info(
                        "post-seed position backfill scan_id=%s stats=%s",
                        scan_id, stats,
                    )
                except Exception as exc:
                    logger.warning(
                        "post-seed position backfill 실패 scan_id=%s err=%s",
                        scan_id, exc,
                    )
            except Exception as exc:
                logger.error(
                    "rtabmap_seeder: failed scan_id=%s err=%s", scan_id, exc,
                )
                async with AsyncSession(self._engine) as session:
                    async with session.begin():
                        await BuildJobRepository(session).update_state(
                            build_job_id=build_job_id,
                            state=BuildState.FAILED,
                            step=BuildStep.QUALITY_GATE,
                            progress=0.2,
                            failure_reason=BuildFailureReason.INTERNAL,
                            failure_detail={"error": f"rtabmap_seed_failed: {exc}"},
                        )
                raise

        # baseline polygon: dense_video_floor evidence 에서 가져옴 (없으면 None)
        baseline = self._dense_video_floor_baseline(scan_id, build_job_id)

        out_dir = settings.storage_root / "builds" / build_job_id / "sprint82"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = await asyncio.to_thread(
                run_sprint82,
                merged_db=merged_db,
                baseline_polygon=baseline,
                out_dir=out_dir,
            )
        except Exception as exc:
            logger.error(
                "video_mode_pipeline: pipeline 실패 scan_id=%s err=%s",
                scan_id, exc,
            )
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    await BuildJobRepository(session).update_state(
                        build_job_id=build_job_id,
                        state=BuildState.FAILED,
                        step=BuildStep.QUALITY_GATE,
                        progress=0.5,
                        failure_reason=BuildFailureReason.INTERNAL,
                        failure_detail=str(exc)[:2000],
                        finished_at=datetime.now(tz=UTC),
                    )
            return

        nodes_out = result.get("nodes", [])
        edges_out = result.get("edges", [])
        nav_metrics = result.get("metrics", {}).get("navgraph", {})

        # quality gate: nodes≥3, edges≥nodes-1
        n_nodes = len(nodes_out)
        n_edges = len(edges_out)
        if n_nodes < 3 or n_edges < max(0, n_nodes - 1):
            detail = (
                f"video_mode_navgraph_too_small: nodes={n_nodes} edges={n_edges}. "
                f"min nodes=3, min edges={max(0, n_nodes - 1)}"
            )
            logger.warning("video_mode_pipeline: quality gate FAIL scan_id=%s %s", scan_id, detail)
            async with AsyncSession(self._engine) as session:
                async with session.begin():
                    await BuildJobRepository(session).update_state(
                        build_job_id=build_job_id,
                        state=BuildState.FAILED,
                        step=BuildStep.QUALITY_GATE,
                        progress=0.9,
                        failure_reason=BuildFailureReason.INTERNAL,
                        failure_detail=detail,
                        finished_at=datetime.now(tz=UTC),
                    )
            return

        logger.info(
            "video_mode_pipeline: navgraph nodes=%d edges=%d scan_id=%s",
            n_nodes, n_edges, scan_id,
        )

        # MapNodeVO / MapEdgeVO 변환
        from uuid import NAMESPACE_URL, uuid5
        scan_uuid = UUID(scan_id)
        job_uuid = UUID(build_job_id)

        def _make_node(raw: dict[str, Any]) -> MapNodeVO:
            nid_str = str(raw.get("node_id", raw.get("id", "")))
            try:
                nid = UUID(nid_str)
            except ValueError:
                nid = uuid5(NAMESPACE_URL, f"{build_job_id}/{nid_str}")
            return MapNodeVO(
                node_id=nid,
                scan_id=scan_uuid,
                build_job_id=job_uuid,
                node_type=NodeType.CORRIDOR,
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                z=0.0,
            )

        def _make_edge(
            raw: dict[str, Any],
            node_coords: dict[UUID, tuple[float, float]],
        ) -> MapEdgeVO | None:
            fid_str = str(raw.get("from_id", raw.get("from", "")))
            tid_str = str(raw.get("to_id", raw.get("to", "")))
            try:
                fid = UUID(fid_str) if fid_str else None
                tid = UUID(tid_str) if tid_str else None
            except ValueError:
                # uuid5 fallback
                fid = uuid5(NAMESPACE_URL, f"{build_job_id}/{fid_str}") if fid_str else None
                tid = uuid5(NAMESPACE_URL, f"{build_job_id}/{tid_str}") if tid_str else None
            if fid is None or tid is None:
                return None
            # polyline: from/to 노드 좌표로 최소 2점 linestring 구성
            fx, fy = node_coords.get(fid, (0.0, 0.0))
            tx, ty = node_coords.get(tid, (0.0, 0.0))
            polyline: list[tuple[float, float, float]] = [(fx, fy, 0.0), (tx, ty, 0.0)]
            return MapEdgeVO(
                edge_id=uuid4(),
                scan_id=scan_uuid,
                build_job_id=job_uuid,
                from_node_id=fid,
                to_node_id=tid,
                edge_type=EdgeType.SKELETON,
                polyline=polyline,
                length_m=float(raw.get("weight", 1.0)),
            )

        map_nodes = [_make_node(n) for n in nodes_out]
        node_coords = {n.node_id: (n.x, n.y) for n in map_nodes}
        map_edges = [
            e for raw in edges_out
            if (e := _make_edge(raw, node_coords)) is not None
        ]

        # A-1/A-2: POI projection step — sprint 49 모듈 재사용
        # video_mode path에서도 POIProjectionStep을 inline 호출한다.
        async with AsyncSession(self._engine) as _poi_session:
            pois = await self._load_pois(_poi_session, scan_id)
            keyframes = await self._load_keyframes(_poi_session, scan_id)

        if pois:
            try:
                from indoor_server.application.building.arkit_to_rtabmap_transform import (
                    EstimateParams,
                    build_pairs_from_keyframes_and_nodes,
                    estimate_arkit_to_rtabmap_transform,
                )
                from indoor_server.application.building.steps.poi_projection import (
                    POIProjectionStep,
                )

                # RTABMap node poses (merged_db로부터 읽음)
                rtabmap_node_pose_by_id: dict[int, tuple[float, float, float]] = {}
                try:
                    from indoor_server.application.rtabmap.reader import RtabmapReader
                    rn = await asyncio.to_thread(
                        RtabmapReader().load_nodes, merged_db
                    )
                    # RtabmapNode pose Matrix4x4: translation은 [0][3],[1][3],[2][3]
                    rtabmap_node_pose_by_id = {
                        n.node_id: (
                            float(n.pose[0][3]),
                            float(n.pose[1][3]),
                            float(n.pose[2][3]),
                        )
                        for n in rn
                    }
                except Exception as _rn_err:
                    logger.warning(
                        "video_mode_pipeline: RTABMap node load 실패 "
                        "— transform 없이 POI projection: %s",
                        _rn_err,
                    )

                keyframe_pose_by_node_id: dict[int, tuple[float, float, float]] = {
                    kf.rtabmap_node_id: (kf.tx, kf.ty, kf.tz)
                    for kf in keyframes
                    if kf.rtabmap_node_id is not None
                }
                pairs = build_pairs_from_keyframes_and_nodes(
                    keyframe_pose_by_rtabmap_node_id=keyframe_pose_by_node_id,
                    rtabmap_node_pose_by_id=rtabmap_node_pose_by_id,
                )
                result_tf = estimate_arkit_to_rtabmap_transform(
                    pairs, params=EstimateParams()
                )
                tf = result_tf.transform

                step = POIProjectionStep(
                    scan_id=scan_uuid,
                    build_job_id=job_uuid,
                    arkit_to_rtabmap_transform=tf,
                )
                # A-2: label None이면 "301호" 하드코딩 (ingest 시 label 없는 경우 대비)
                filled_pois = []
                for p in pois:
                    if not p.label:
                        p = p.model_copy(update={"label": "301호"})
                    filled_pois.append(p)

                map_nodes, map_edges, _poi_poses = step.run(
                    filled_pois, map_nodes, map_edges
                )
                logger.info(
                    "video_mode_pipeline: POI projection done poi_count=%d transform_confidence=%s",
                    len(filled_pois),
                    tf.confidence,
                )
            except Exception as _poi_err:
                logger.warning(
                    "video_mode_pipeline: POI projection 실패 — POI 없이 build 계속: %s",
                    _poi_err,
                )

        # branch_mark: 사용자가 찍은 분기점 → graph 의 junction 노드로 추가.
        # 좌표는 ARKit Y-up frame 이므로 transform 적용 (POI 와 동일 방식).
        try:
            async with AsyncSession(self._engine) as _br_session:
                from indoor_server.infrastructure.db import tables as _t
                import sqlalchemy as _sa
                br_rows = (
                    await _br_session.execute(
                        _sa.select(
                            _t.branch_mark.c.id,
                            _t.branch_mark.c.tx,
                            _t.branch_mark.c.ty,
                            _t.branch_mark.c.tz,
                        ).where(_t.branch_mark.c.scan_id == scan_id)
                    )
                ).fetchall()
            if br_rows and pois:
                # tf 는 위 POI 처리에서 만든 transform 재사용
                from indoor_server.domain.building.enums import NodeType as _NT
                br_count = 0
                for br in br_rows:
                    arkit_xyz = (float(br.tx), float(br.ty), float(br.tz))
                    if tf is not None and tf.confidence == "high":
                        wx, wy, wz = tf.apply(arkit_xyz)
                    else:
                        wx, wy, wz = arkit_xyz
                    br_node = MapNodeVO(
                        node_id=uuid5(job_uuid, f"branch-mark:{int(br.id)}"),
                        scan_id=scan_uuid,
                        build_job_id=job_uuid,
                        node_type=_NT.JUNCTION,
                        x=float(wx), y=float(wy), z=float(wz),
                        label=None,
                        source_ref={
                            "role": "branch_mark",
                            "branch_mark_id": int(br.id),
                            "raw_arkit_xyz": list(arkit_xyz),
                        },
                    )
                    map_nodes.append(br_node)
                    # 가장 가까운 corridor/junction 노드와 spur edge
                    target = min(
                        (n for n in map_nodes if n.node_type in (
                            NodeType.CORRIDOR, NodeType.POI_ATTACH, NodeType.JUNCTION
                        ) and n.node_id != br_node.node_id),
                        key=lambda n: math.hypot(n.x - br_node.x, n.y - br_node.y),
                        default=None,
                    )
                    if target is not None:
                        d = math.hypot(target.x - br_node.x, target.y - br_node.y)
                        if d > 0.001:
                            map_edges.append(MapEdgeVO(
                                edge_id=uuid5(job_uuid, f"branch-edge:{int(br.id)}"),
                                scan_id=scan_uuid,
                                build_job_id=job_uuid,
                                from_node_id=br_node.node_id,
                                to_node_id=target.node_id,
                                edge_type=EdgeType.SKELETON,
                                polyline=[(br_node.x, br_node.y, br_node.z),
                                          (target.x, target.y, target.z)],
                                length_m=d,
                            ))
                    br_count += 1
                logger.info(
                    "video_mode_pipeline: branch nodes 추가 count=%d scan_id=%s",
                    br_count, scan_id,
                )
        except Exception as _br_err:
            logger.warning(
                "video_mode_pipeline: branch_mark 통합 실패 scan_id=%s err=%s",
                scan_id, _br_err,
            )

        # A-6: interfloor_mark connector nodes 통합 (기존 경로와 동일 헬퍼 사용)
        if interfloor_marks:
            map_nodes, map_edges = _append_interfloor_connector_nodes(
                interfloor_marks=interfloor_marks,
                nodes=map_nodes,
                edges=map_edges,
                scan_id=scan_uuid,
                build_job_id=job_uuid,
            )

        # A-6b: mock vertical_connector catalog 행도 graph에 passage 노드로 추가.
        # interfloor_mark가 없는 경우(mock-only scan)에도 동작하도록 별도 분기.
        # production 가드: settings.enable_mock_passage가 OFF면 mock 노드 주입 금지.
        from indoor_server.config import settings as _build_settings
        mock_connectors: list[Any] = []
        if _build_settings.enable_mock_passage:
            async with AsyncSession(self._engine) as _mc_session:
                mock_connectors = await _load_mock_connectors_for_scan(
                    session=_mc_session,
                    scan_id=scan_id,
                )
        if mock_connectors:
            map_nodes, map_edges = _append_mock_connector_nodes(
                mock_connectors=mock_connectors,
                nodes=map_nodes,
                edges=map_edges,
                scan_id=scan_uuid,
                build_job_id=job_uuid,
            )
            logger.info(
                "video_mode_pipeline: mock connector nodes 추가 count=%d scan_id=%s",
                len(mock_connectors), scan_id,
            )

        # PERSIST
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                await BuildJobRepository(session).update_progress(
                    build_job_id=build_job_id,
                    step=BuildStep.PERSIST,
                    progress=0.97,
                )
                await MapGraphRepository(session).replace_graph(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                    nodes=map_nodes,
                    edges=map_edges,
                )
                await POICatalogService(session).sync_scan_pois(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                )
                await VerticalConnectorCatalogService(session).sync_scan_interfloor_marks(
                    scan_id=scan_id,
                    build_job_id=build_job_id,
                )
                counts = BuildCounts(
                    map_nodes=len(map_nodes),
                    map_edges=len(map_edges),
                    build_source="video_mode_sprint82",
                    rtabmap=nav_metrics,
                )
                await BuildJobRepository(session).update_state(
                    build_job_id=build_job_id,
                    state=BuildState.SUCCEEDED,
                    step=BuildStep.DONE,
                    progress=1.0,
                    counts=counts,
                    finished_at=datetime.now(tz=UTC),
                )

        logger.info(
            "video_mode_pipeline: SUCCEEDED scan_id=%s nodes=%d edges=%d",
            scan_id, len(map_nodes), len(map_edges),
        )

        # Server 의 SuperPoint cache 를 미리 채움 (cold-start 제거).
        # fire-and-forget — server 가 background thread 로 indexing.
        try:
            await self._warmup_superpoint(
                scan_id=scan_id,
                rtabmap_db_path=settings.storage_root / "scans" / scan_id / "rtabmap.db",
            )
        except Exception as exc:
            logger.warning("superpoint warmup trigger 실패 scan_id=%s err=%s", scan_id, exc)

    async def _warmup_superpoint(self, *, scan_id: str, rtabmap_db_path: Path) -> None:
        """Build 완료 후 server 의 SuperPoint cache 를 preload."""
        async with AsyncSession(self._engine) as session:
            # scan_id → floor_id (warmup 시 SuperPoint engine 의 map_id 가 floor_id 임)
            row = (
                await session.execute(
                    sa.select(t.floor_scan.c.floor_id)
                    .where(t.floor_scan.c.scan_id == scan_id)
                    .where(t.floor_scan.c.active == sa.true())
                    .limit(1)
                )
            ).first()
        if row is None:
            return
        floor_id = str(row.floor_id)
        # server 컨테이너 internal port 8000 (compose service name=server)
        import httpx as _httpx
        url = "http://server:8000/admin/superpoint/warmup"
        payload = {"map_id": floor_id, "db_path": str(rtabmap_db_path)}
        async with _httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            logger.info(
                "superpoint warmup queued floor_id=%s status=%d",
                floor_id, resp.status_code,
            )

    def _dense_video_floor_baseline(self, scan_id: str, build_job_id: str) -> Path | None:
        """dense_video_floor evidence 에서 baseline polygon 경로를 반환. 없으면 None."""
        try:
            from indoor_server.config import settings
            candidate = (
                settings.debug_root / scan_id / build_job_id
                / "dense_video_floor" / "floor_polygon.geojson"
            )
            if candidate.exists():
                return candidate
        except Exception:
            pass
        return None

    async def _fail_rtabmap_not_ready(
        self,
        *,
        scan_id: str,
        build_job_id: str,
        diagnostics: RtabmapDiagnostics,
    ) -> None:
        counts = BuildCounts(
            build_source="rtabmap_required",
            rtabmap=diagnostics.to_counts_dict(),
        )
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                await BuildJobRepository(session).update_state(
                    build_job_id=build_job_id,
                    state=BuildState.FAILED,
                    step=BuildStep.QUALITY_GATE,
                    progress=0.05,
                    failure_reason=BuildFailureReason.RTABMAP_DATA_NOT_READY,
                    failure_detail=";".join(diagnostics.issues)[:2000],
                    counts=counts,
                    finished_at=datetime.now(tz=UTC),
                )
        logger.warning(
            "build failed before legacy map path scan_id=%s build_job_id=%s "
            "rtabmap_issues=%s",
            scan_id,
            build_job_id,
            diagnostics.issues,
        )


# ── Sprint 84 interfloor connector helpers ───────────────────────────────────


def _append_interfloor_connector_nodes(
    *,
    interfloor_marks: list[InterfloorMarkDbRow],
    nodes: list[MapNodeVO],
    edges: list[MapEdgeVO],
    scan_id: UUID,
    build_job_id: UUID,
) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
    """Create connector route nodes without writing synthetic negative POI FKs."""
    if not interfloor_marks:
        return nodes, edges

    base_nodes = list(nodes)
    out_nodes = list(nodes)
    out_edges = list(edges)
    for mark in interfloor_marks:
        connector_type = mark.connector_type.strip().lower()
        connector_key = _connector_key_from_prefix(mark.prefix)
        label = f"{connector_type.upper()} {mark.prefix}"
        connector_node = MapNodeVO(
            node_id=uuid5(build_job_id, f"interfloor-connector:{mark.id}"),
            scan_id=scan_id,
            build_job_id=build_job_id,
            node_type=NodeType.POI,
            x=mark.tx,
            y=mark.ty,
            z=mark.tz,
            label=label,
            poi_mark_id=None,
            source_ref={
                "role": "vertical_connector_stop",
                "interfloor_mark_id": mark.id,
                "connector_type": connector_type,
                "connector_key": connector_key,
                "prefix": mark.prefix,
                "keyframe_seq": mark.keyframe_seq,
                "raw_arkit_xyz": [mark.tx, mark.ty, mark.tz],
            },
        )
        out_nodes.append(connector_node)
        attach = _nearest_connector_attach_node(connector_node, base_nodes)
        if attach is None:
            continue
        length_m = _distance_3d(connector_node, attach)
        if length_m <= 0.001:
            continue
        out_edges.append(
            MapEdgeVO(
                edge_id=uuid5(build_job_id, f"interfloor-connector-edge:{mark.id}"),
                scan_id=scan_id,
                build_job_id=build_job_id,
                from_node_id=connector_node.node_id,
                to_node_id=attach.node_id,
                edge_type=EdgeType.POI_SPUR,
                polyline=[
                    (connector_node.x, connector_node.y, connector_node.z),
                    (attach.x, attach.y, attach.z),
                ],
                length_m=length_m,
            )
        )
    return out_nodes, out_edges


def _nearest_connector_attach_node(
    connector_node: MapNodeVO,
    nodes: list[MapNodeVO],
) -> MapNodeVO | None:
    candidates = [
        node for node in nodes
        if (node.source_ref or {}).get("role") != "vertical_connector_stop"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda node: _distance_2d(connector_node, node))


def _distance_2d(a: MapNodeVO, b: MapNodeVO) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _distance_3d(a: MapNodeVO, b: MapNodeVO) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _connector_key_from_prefix(prefix: str) -> str:
    value = prefix.strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return out or "default"


# ── Sprint 78: mock vertical_connector catalog → graph 노드 통합 ───────────


async def _load_mock_connectors_for_scan(
    *,
    session: AsyncSession,
    scan_id: str,
) -> list[dict[str, Any]]:
    """scan_id → floor → building 경로로 is_mock=true vertical_connector 행을 조회한다."""
    from indoor_server.infrastructure.db import tables as _t

    result = await session.execute(
        sa.select(
            _t.vertical_connector.c.connector_id,
            _t.vertical_connector.c.connector_type,
            _t.vertical_connector.c.connector_key,
            _t.vertical_connector.c.name,
        )
        .select_from(
            _t.floor_scan.join(
                _t.building_floor,
                _t.floor_scan.c.floor_id == _t.building_floor.c.floor_id,
            ).join(
                _t.vertical_connector,
                _t.vertical_connector.c.building_id == _t.building_floor.c.building_id,
            )
        )
        .where(
            _t.floor_scan.c.scan_id == scan_id,
            _t.vertical_connector.c.is_mock == sa.true(),
        )
    )
    return [dict(row._mapping) for row in result.fetchall()]


def _append_mock_connector_nodes(
    *,
    mock_connectors: list[dict[str, Any]],
    nodes: list[MapNodeVO],
    edges: list[MapEdgeVO],
    scan_id: UUID,
    build_job_id: UUID,
) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
    """is_mock vertical_connector 행을 graph에 passage 노드로 추가한다.

    mock connector는 실제 pose가 없으므로 현재 그래프의 centroid를 배치 위치로 사용한다.
    """
    if not mock_connectors or not nodes:
        return nodes, edges

    # 그래프 centroid 계산 (connector_stop이 아닌 노드 기준)
    base_nodes = [
        n for n in nodes
        if (n.source_ref or {}).get("role") != "vertical_connector_stop"
    ]
    if not base_nodes:
        return nodes, edges

    cx = sum(n.x for n in base_nodes) / len(base_nodes)
    cy = sum(n.y for n in base_nodes) / len(base_nodes)

    out_nodes = list(nodes)
    out_edges = list(edges)

    for mc in mock_connectors:
        connector_type = str(mc.get("connector_type", "elevator")).lower()
        connector_key = str(mc.get("connector_key", "unknown")).lower()
        connector_id = str(mc.get("connector_id", ""))
        label = str(mc.get("name") or f"{connector_type.upper()} {connector_key}")

        connector_node = MapNodeVO(
            node_id=uuid5(build_job_id, f"mock-connector:{connector_id}"),
            scan_id=scan_id,
            build_job_id=build_job_id,
            node_type=NodeType.POI,
            x=cx,
            y=cy,
            z=0.0,
            label=label,
            poi_mark_id=None,
            source_ref={
                "role": "vertical_connector_stop",
                "connector_type": connector_type,
                "connector_key": connector_key,
                "connector_id": connector_id,
                "is_mock": True,
            },
        )
        out_nodes.append(connector_node)

        attach = _nearest_connector_attach_node(connector_node, base_nodes)
        if attach is None:
            continue
        length_m = _distance_2d(connector_node, attach)
        if length_m <= 0.001:
            continue
        out_edges.append(
            MapEdgeVO(
                edge_id=uuid5(build_job_id, f"mock-connector-edge:{connector_id}"),
                scan_id=scan_id,
                build_job_id=build_job_id,
                from_node_id=connector_node.node_id,
                to_node_id=attach.node_id,
                edge_type=EdgeType.POI_SPUR,
                polyline=[
                    (connector_node.x, connector_node.y, connector_node.z),
                    (attach.x, attach.y, attach.z),
                ],
                length_m=length_m,
            )
        )

    return out_nodes, out_edges
