"""처음 보는 물건용 치수 회귀 모델 학습.

API와 똑같이 이미지에서 배경 차분으로 물체 영역을 따서 학습한다 (라벨 폴리곤으로 학습하면 분할 오차를 못 배움).
- 이미지: Validation 원천(VS.zip)을 zip에서 바로 읽음 (압축 해제 불필요)
- 정답: Validation 라벨(VL.zip)의 실측 치수 (내림차순)
- 입력: cam 1~3 영역 기하 특징 21개 (core.features.shape_features x 3)
- 모델: 치수 3개 각각 HistGradientBoosting (log 공간)  → outputs/volume_model.joblib
- 배경(build_background.py)을 먼저 만들어야 한다.

사용법: python scripts/train_volume_model.py [--vs-zip 경로] [--vl-zip 경로] [--workers 8]
"""
import argparse
import io
import json
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from _paths import OUT_DIR, VL_ZIP, VS_ZIP

_worker = {}


def zip_names(path):
    """정규화된 한글 경로 -> zip 내부 원래 이름"""
    out = {}
    for i in zipfile.ZipFile(path).infolist():
        n = i.filename
        if not (i.flag_bits & 0x800):
            try:
                n = n.encode("cp437").decode("cp949")
            except UnicodeError:
                pass
        out[unicodedata.normalize("NFC", n)] = i.filename
    return out


def _init(vs_zip):
    from core.segment import BackgroundSegmenter
    _worker["zip"] = zipfile.ZipFile(vs_zip)
    _worker["seg"] = BackgroundSegmenter(OUT_DIR)


def _features(job):
    from PIL import Image
    from core.features import shape_features
    raw, cam = job
    image = Image.open(io.BytesIO(_worker["zip"].read(raw))).convert("RGB")
    try:
        return shape_features(_worker["seg"](image, cam))
    except ValueError:  # 물체를 못 찾은 이미지는 제외
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs-zip", default=str(VS_ZIP))
    parser.add_argument("--vl-zip", default=str(VL_ZIP))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    images, labels = zip_names(args.vs_zip), zip_names(args.vl_zip)
    vl = zipfile.ZipFile(args.vl_zip)
    jobs, meta = [], []
    for name, raw in images.items():
        if not (name.startswith("01_입고물품") and name.endswith(".jpg")) or name[:-4] + ".json" not in labels:
            continue
        at = json.loads(vl.read(labels[name[:-4] + ".json"]))["annotations"][0]["attributes"]
        dims = sorted([at["length"], at["width"], at["height"]], reverse=True)
        if min(dims) <= 0:
            continue
        jobs.append((raw, int(name[-5])))
        meta.append((name.split("/")[-2], int(name[-7]), int(name[-5]), dims))
    print(f"이미지 {len(jobs)}장 배경 차분 중...", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.vs_zip,)) as ex:
        feats = list(ex.map(_features, jobs, chunksize=16))

    groups = {}
    for (prod, shot, cam, dims), f in zip(meta, feats):
        if f is not None:
            groups.setdefault((prod, shot), {"dims": dims})[cam] = f
    groups = [g for g in groups.values() if all(c in g for c in (1, 2, 3))]
    X = np.array([g[1] + g[2] + g[3] for g in groups])
    y = np.log([g["dims"] for g in groups])

    model = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0))
    model.fit(X, y)
    out = OUT_DIR / "volume_model.joblib"
    joblib.dump(model, out)
    print(f"학습 샘플 {len(groups)}개 (상품 x shot), 저장: {out}")


if __name__ == "__main__":
    main()
