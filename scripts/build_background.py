"""cam별 '빈 바닥' 배경 이미지 만들기.

고정 카메라라서 서로 다른 물건 사진 여러 장의 픽셀별 중앙값 = 물건이 없는 바닥.
(실제 현장이라면 빈 컨베이어를 한 장 찍어 두면 된다.)

사용법: python scripts/build_background.py [--sample-root data/Sample]
"""
import argparse
import glob
import json

import numpy as np
from PIL import Image

from _paths import OUT_DIR, SAMPLE_ROOT
from core.segment import BG_SCALE

parser = argparse.ArgumentParser()
parser.add_argument("--sample-root", default=str(SAMPLE_ROOT))
args = parser.parse_args()

rois = {}
for cam in (1, 2, 3):
    paths = sorted(glob.glob(f"{args.sample_root}/01.원천데이터/01_입고물품/**/*_{cam}.jpg", recursive=True))

    # 물체가 놓이는 바닥 영역(ROI) = 라벨 bbox들의 합집합 + 여유 10%
    boxes = []
    for p in paths:
        x, y, w, h = json.load(open(p.replace("01.원천데이터", "02.라벨링데이터")[:-4] + ".json", encoding="utf-8"))["annotations"][0]["bbox"]
        boxes.append([x, y, x + w, y + h])
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    rois[cam] = [max(0, x0 - mx), max(0, y0 - my), min(1920, x1 + mx), min(1080, y1 + my)]

    stack = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        stack.append(np.asarray(img.resize((round(img.width * BG_SCALE), round(img.height * BG_SCALE)), Image.BILINEAR)))
    Image.fromarray(np.median(np.stack(stack), axis=0).astype(np.uint8)).save(OUT_DIR / f"background_cam{cam}.png")
    print(f"cam{cam}: {len(paths)}장 중앙값 → {OUT_DIR / f'background_cam{cam}.png'}, ROI {[round(v) for v in rois[cam]]}")

(OUT_DIR / "roi.json").write_text(json.dumps(rois))
