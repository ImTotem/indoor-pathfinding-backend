# Colab GPU 학습 가이드 (Sprint 57)

Mac 발열 없이 학습. 출력은 `best.pt` 한 파일 — 로컬 `infer.py`에서 그대로 사용 가능.

## 흐름

```
[Mac]                                    [Colab T4]                     [Mac]
1. zip 패키지 만들기  ─→  2. zip 업로드 + 학습 (~12분)  ─→  3. best.pt 다운로드
                                                                        ↓
                                                              4. python infer.py
```

## 1. zip 패키지 만들기 (로컬)

```bash
cd server/scripts/wall_polygon_dl
bash make_colab_zip.sh
# → 출력: ../../../_workspace/sprint_57_neural_polygon/wall_polygon_dl_colab.zip
```

zip 내용물:
- `data_generator.py`
- `dataset.py`
- `model.py`
- `train.py`
- `cache_v1/000000.npz ... 000999.npz` (1000 sample 미리 생성된 페어)

## 2. Colab 실행

1. <https://colab.research.google.com> 접속 (Google 계정 로그인)
2. **File → Upload notebook** → `train_colab.ipynb` 선택
3. **Runtime → Change runtime type → T4 GPU** 선택, Save
4. 좌측 Files 📁 패널 → 업로드 버튼 → `wall_polygon_dl_colab.zip` 선택
5. 위에서부터 셀 순서대로 실행 (각 셀 좌측 ▶ 클릭)
   - cell 0: GPU 확인
   - cell 1: 패키지 설치
   - cell 2: zip extract
   - cell 3: 학습 (~12분)
   - cell 4: best.pt 다운로드

## 3. 로컬 inference

```bash
cd server/scripts/wall_polygon_dl
source .venv/bin/activate
python infer.py \
  ../../../_workspace/sprint_55_heatmap_boundary_display_graph/evidence/56A8698C_boundary_graph_v7_final/obstacle_heatmap_counts.npz \
  --checkpoint ~/Downloads/best.pt \
  --out-dir ../../../_workspace/sprint_57_neural_polygon/infer_colab \
  --device mps
```

## 학습 spec 조절

`train_colab.ipynb` 3번 셀에서:

```python
EPOCHS = 30      # 늘리면 정밀도 ↑, 시간 ↑
BATCH = 8        # T4 16GB이면 8 OK, OOM 나면 4로 줄이기
RESIZE = 512     # 256→384→512 디테일 증가, 시간/메모리 증가
ENCODER = 'resnet34'  # 'resnet50' 더 정밀, 'mobilenet_v2' 빠름
```

추천:
- **PoC 재실행 (검증)**: 위 default 그대로 (~12분 / val IoU ≥ 0.95 예상)
- **Production**: `RESIZE=512, ENCODER='resnet50', EPOCHS=50` (~30분)

## Tip

- **12시간 disconnect**: `runs/best.pt`는 매 epoch 갱신되니 중간에 끊겨도 가장 좋은 모델은 보존됨. 학습 도중 best.pt 다운로드 가능.
- **Drive 마운트**: notebook 마지막 셀 — best.pt를 Drive에 자동 저장. 다음 세션 재학습 시 cache 재사용 가능.
- **무료 tier 한도**: 같은 계정으로 하루 ~12시간. 초과 시 다음날까지 GPU 못 받음. PoC 1번이면 무관.
