# wall_polygon_dl — Sprint 57 PoC

Synthetic-data U-Net for `obstacle heatmap → building footprint mask`.

## 0. Setup (one-time)

서버 prod deps와 분리. M4 Pro 권장.

```bash
cd server/scripts/wall_polygon_dl
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Sanity check (~30 sec)

```bash
python sanity_check.py --n 20 --out-dir ../../../_workspace/sprint_57_neural_polygon/sanity
```

→ generator가 20 style 모두 정상 생성하는지, real heatmap과 분포 비슷한지 시각 확인.

## 2. PoC train (M4 MPS, 30 epochs ≈ 30~60분 예상)

```bash
python train.py \
  --train 900 --val 100 \
  --epochs 30 --batch 4 --lr 1e-3 \
  --resize-to 512 --device mps \
  --cache-dir ../../../_workspace/sprint_57_neural_polygon/cache \
  --out ../../../_workspace/sprint_57_neural_polygon/runs/poc_v1
```

`--resize-to 512`로 700×700 input을 학습 시 다운스케일 (M4 Pro 메모리 안전).
`--cache-dir`로 sample을 디스크에 저장해 재실행 가속.

## 3. Real RTABMap heatmap inference

```bash
python infer.py \
  ../../../_workspace/sprint_55_heatmap_boundary_display_graph/evidence/56A8698C_boundary_graph_v7_final/obstacle_heatmap_counts.npz \
  --checkpoint ../../../_workspace/sprint_57_neural_polygon/runs/poc_v1/best.pt \
  --out-dir ../../../_workspace/sprint_57_neural_polygon/infer_56A8698C \
  --device mps \
  --strip-purple \
  --purple-g-threshold 60 \
  --rectify \
  --rectify-grid-m 0.10 \
  --polygon-mode image700
```

산출:
- `input_inferno.png` — 모델에 들어가는 inferno-style heatmap
- `input_after_strip.png` — Sprint 60 방식 `G<60` 보라색 smear 제거 입력
- `predicted_mask.png` — building footprint mask
- `overlay.png` — input/mask, rectified polygon, graph 3-panel 시각화
- `polygon.geojson` — `image700` mode에서는 local metric polygon (`0.05m/px`)
- `graph.geojson` — Sprint 59 path corridor graph

주의: Sprint 60과 같은 결과를 재현하려면 heatmap source도 기존과 같아야 한다.
실 scan `56A8698C` 기준으로는 `run_real_wall_polygon_evidence.py`를
`--obstacle-mask-source inverse_floor`로 실행해 생성한
`obstacle_heatmap_counts.npz`가 Sprint 55/57 입력과 동일하다.

## 4. Acceptance

- synthetic val IoU ≥ 0.85
- real scan polygon이 T자 corridor 형태 cover (사용자 시각 인정)
- vertices ≤ 80
- inference < 1s (M4 MPS)

## 5. Production gate (PoC 통과 후)

- 10k sample + augmentation
- real scan 5개 수동 라벨 → fine-tune
- ONNX export → server step 통합
