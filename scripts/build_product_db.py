"""'갖고 있는 물건' 상품 DB 만들기 (Sample 입고 데이터).

상품당 shot 1·2 x cam 1~3 = 6장을 DINOv2-small 임베딩으로 저장하고, 라벨 실측값(내림차순)을 함께 저장한다.
기본은 Sample 40종 전부 등록. 처음 보는 물건을 테스트하려면 --holdout N 으로 N개 상품을 빼고 등록한다
(뺀 목록은 outputs/holdout.json, 전부 등록이면 빈 목록).
배경(build_background.py)을 먼저 만들어야 한다.

사용법: python scripts/build_product_db.py [--sample-root data/Sample] [--holdout 0]
"""
import argparse
import glob
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _paths import OUT_DIR, SAMPLE_ROOT
from core import geometry
from core.features import Embedder
from core.segment import BackgroundSegmenter

parser = argparse.ArgumentParser()
parser.add_argument("--sample-root", default=str(SAMPLE_ROOT))
parser.add_argument("--holdout", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

labels = sorted(glob.glob(f"{args.sample_root}/02.라벨링데이터/01_입고물품/**/*.json", recursive=True))
by_product = {}
for f in labels:
    by_product.setdefault(unicodedata.normalize("NFC", Path(f).parent.name), []).append(f)

names = sorted(by_product)
holdout = {str(n) for n in np.random.default_rng(args.seed).choice(names, args.holdout, replace=False)} if args.holdout else set()
embed = Embedder(torch.device("cpu"))
segment = BackgroundSegmenter(OUT_DIR)

products, embeddings, index = [], [], []
for name in names:
    if name in holdout:
        continue
    crops, info = [], None
    for f in by_product[name]:
        d = json.load(open(f, encoding="utf-8"))
        a = d["annotations"][0]
        image = Image.open(f.replace("02.라벨링데이터", "01.원천데이터").removesuffix(".json") + ".jpg").convert("RGB")
        pts, _ = segment(image, d["images"][0]["cam_id"])   # 조회 때와 같은 방식(배경 차분)으로 크롭
        crops.append(geometry.crop_square(image, geometry.bbox(pts)))
        at = a["attributes"]
        info = {"kan_code": at["KAN_code"], "name": at["product_name"], "barcode": at["barcode"],
                "dims_cm": sorted([at["length"], at["width"], at["height"]], reverse=True)}
    embeddings.append(embed(crops))
    index += [len(products)] * len(crops)
    products.append(info)
    print(f"[{len(products):2d}] {info['name']}  {info['dims_cm']}")

np.savez(OUT_DIR / "product_db.npz", embeddings=np.concatenate(embeddings), product_index=np.array(index))
(OUT_DIR / "products.json").write_text(json.dumps(products, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT_DIR / "holdout.json").write_text(json.dumps(sorted(holdout), ensure_ascii=False, indent=1), encoding="utf-8")
print(f"등록 {len(products)}종 ({len(index)}장), 미등록(테스트용) {len(holdout)}종: {sorted(holdout)}")
