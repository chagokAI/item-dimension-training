# item-dimension-training

Scripts to build the background, product DB, and volume model for [item-dimension-api](https://github.com/chagokAI/item-dimension-api).

[item-dimension-api](https://github.com/chagokAI/item-dimension-api)(입고 물품 사진 → 상품 식별 + 치수·부피 반환)가 추론에 쓰는 **모델·데이터 파일을 만드는 레포**입니다.
이 레포만으로 독립 실행되며, 결과물(`outputs/`)을 API 레포의 `app/data/`로 복사해서 사용합니다.

---

## 1. 서빙에서의 역할

API는 고정 카메라 3대(cam1~3)로 찍은 입고 물품 사진 3장을 받아 아래 순서로 처리합니다.
이 레포는 각 단계에 필요한 파일을 만듭니다.

```
cam1~3 이미지 (1920x1080)
 │
 ├─ ① 물체 영역 찾기 (배경 차분) ─────────── background_cam{1,2,3}.png, roi.json   ← build_background.py
 │
 ├─ ② 상품 식별 (DINOv2 임베딩 유사도) ───── product_db.npz, products.json        ← build_product_db.py
 │     ├ 유사도 ≥ 0.80 → 등록 물품: products.json의 실측 치수 반환
 │     └ 유사도 < 0.80 → 처음 보는 물품 ↓
 │
 └─ ③ 치수 추정 (처음 보는 물품만) ─────────── volume_model.joblib                 ← train_volume_model.py
```

| 결과물 | API에서 쓰이는 곳 | 내용 |
|---|---|---|
| `background_cam{1,2,3}.png` | ① 배경 차분 | cam별 "빈 바닥" 이미지 (960x540). 들어온 사진과 비교해서 달라진 영역을 물체로 판단 |
| `roi.json` | ① 배경 차분 | cam별로 물체가 놓이는 바닥 영역. 벽·커튼 등 화면 가장자리 변화를 무시 |
| `product_db.npz` | ② 상품 식별 | 등록 상품 사진의 DINOv2-small 임베딩 (상품당 6장 x 384차원) |
| `products.json` | ② 상품 식별 → 응답 | 등록 상품 정보: KAN 코드, 상품명, 바코드, **실측 치수(내림차순)** |
| `holdout.json` | (테스트용) | `--holdout N`으로 일부러 등록하지 않은 상품 목록 (기본은 전부 등록이라 빈 목록) |
| `volume_model.joblib` | ③ 치수 추정 | 물체 영역의 기하 특징 → 치수 3개(w ≥ d ≥ h)를 예측하는 회귀 모델 |

---

## 2. 데이터셋

**[AI Hub 물류공간 예측 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71861)** (2024 구축, v1.1)

- 물류센터 입고·출고 물품을 고정 카메라로 촬영한 RGB 이미지 + 라벨(JSON)
- 물품 1개당 6장: **shot 1~2** (물품을 다르게 놓고 2번 촬영) x **cam 1~3** (같은 순간 3방향), 1920x1080
- 라벨: bbox, 폴리곤, KAN 분류코드, 상품명, 바코드, **실측 치수 length·width·height(cm)**, 무게 등
- 이용약관상 재배포 불가 → 이 레포에는 포함하지 않음. 직접 신청해서 받아야 함

### 사용한 부분 (입고물품만 사용, 출고물품은 사용 안 함)

| 데이터 | 규모 | 사용처 | 사용한 라벨 필드 |
|---|---|---|---|
| **Sample** 입고물품 | 40종, 240장 | 배경 이미지, ROI, 등록 상품 DB | `bbox`(ROI 계산), `KAN_code`, `product_name`, `barcode`, `length/width/height` |
| **Validation** 입고물품 (`VS.zip`, `VL.zip`) | 1,014종, 6,049장 → 학습 샘플 1,997개(상품 x shot) | 부피 모델 학습 | `length/width/height` |

- Sample과 Validation은 **겹치는 상품이 없어서**, 부피 모델은 Validation으로 학습하고 Sample로 "처음 보는 물품" 성능을 평가했습니다.
- 등록 상품 DB는 Sample **40종 전부 등록**이 기본값입니다. "처음 보는 물품" 동작을 테스트하려면 `--holdout 5`처럼 일부 상품을 빼고 등록합니다 (뺀 목록은 `holdout.json`).

### 폴더 배치

```
data/                               # git에 올라가지 않음
├─ Sample/
│  ├─ 01.원천데이터/01_입고물품/<대분류>/<중분류>/<상품ID>/<상품ID>_<shot>_<cam>.jpg
│  └─ 02.라벨링데이터/01_입고물품/... (같은 구조, .json)
└─ Validation/
   ├─ 01.원천데이터/VS.zip            # 압축 해제 불필요 (zip에서 바로 읽음)
   └─ 02.라벨링데이터/VL.zip
```

자세한 배치 방법(바로가기로 두는 법 포함)은 [data/README.md](data/README.md) 참고.
다른 위치에 있으면 `DATA_DIR` 환경변수 또는 각 스크립트 인자(`--sample-root`, `--vs-zip`, `--vl-zip`)로 지정합니다.

---

## 3. 실행

아래 순서대로 실행하면 현재 API가 쓰는 결과물과 **같은 파일**이 만들어집니다.

```bash
# 0. 데이터셋을 data/에 배치 (data/README.md 참고)

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/build_background.py     # 1. 배경 + ROI (가장 먼저, 약 10초)
.venv/bin/python scripts/build_product_db.py     # 2. 등록 상품 DB (약 30초, 40종 전부 등록)
.venv/bin/python scripts/train_volume_model.py   # 3. 부피 모델 (CPU 8코어 기준 약 1분)

cp outputs/* ../item-dimension-api/app/data/     # 4. API에 반영 (두 레포가 같은 폴더에 있다고 가정) 후 서버 재시작
```

결과물 8개(`background_cam1~3.png`, `roi.json`, `product_db.npz`, `products.json`, `holdout.json`, `volume_model.joblib`)가 `outputs/`에 생깁니다.

2, 3은 1에서 만든 배경으로 물체 영역을 찾기 때문에, 배경을 다시 만들면 2, 3도 다시 실행해야 합니다.

---

## 4. 방법

| 단계 | 방법 |
|---|---|
| **배경** | cam별 Sample 이미지 80장(40종 x 2 shot)의 픽셀별 중앙값. 서로 다른 물품이 다른 위치에 있어서 중앙값을 취하면 바닥만 남음 |
| **ROI** | Sample 라벨 bbox들의 범위 + 10% 여유 |
| **물체 영역** | ROI 안에서 채널별 밝기를 배경에 맞춘 뒤, 픽셀 차이 > 20인 영역을 정리(열림·닫힘·구멍 메움)하고 가장 큰 덩어리 선택 |
| **등록 상품 DB** | 상품당 6장(세운·눕힌 자세 모두) → 물체 중심 정사각형 크롭(긴 변 x 1.2) → DINOv2-small CLS 임베딩(정규화) |
| **부피 모델** | Validation 이미지를 위와 같은 배경 차분으로 분할 → cam별 기하 특징 7개(bbox 가로·세로, 면적, 회전 최소 사각형 긴·짧은 변, solidity, 종횡비) x 3 = 21개 → log(치수 3개, 내림차순)를 치수별 HistGradientBoosting으로 회귀 |

부피 모델을 라벨 폴리곤이 아닌 **API와 같은 배경 차분 결과로 학습**해서, 분할 오차(그림자, 일부 누락 등)까지 함께 학습합니다.

---

## 5. 성능 (Sample 40종 x 2 shot = 80건, 평가 시 35종 등록 / 5종 미등록으로 나눠 측정)

| 항목 | 결과 |
|---|---|
| 등록 / 처음 봄 판정 | 80/80 |
| 등록 물품 치수 | 라벨 실측값 그대로 (오차 0%) |
| 처음 보는 물품 부피 오차 (Sample 40종 전체 기준) | 중앙값 약 32%, ±30% 이내 약 48% |

처음 보는 물품은 납작한 물건의 두께, 투명한 물건, 흰 바닥과 비슷한 색의 물건에서 오차가 큽니다.

---

## 6. 전제 조건 / 주의

- 데이터셋과 **같은 촬영 환경**(카메라 위치, 바닥, 조명, 1920x1080)에서만 유효합니다. 환경이 바뀌면 배경부터 다시 만들어야 합니다.
- `core/`의 계산(배경 차분, 크롭, 특징 추출)은 item-dimension-api의 `app/segment.py`, `app/geometry.py`, `app/measure.py`와 **같아야** 합니다. 한쪽을 바꾸면 다른 쪽도 같이 수정하고 결과물을 다시 만드세요.
- `outputs/`의 파일은 데이터셋 파생물(상품명, 실측 치수, 배경 이미지)이므로 공개 저장소에 올리지 마세요.
