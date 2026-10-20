"""경로 설정.

- DATA_DIR: AI Hub 데이터셋 위치 (기본: 이 레포의 data/, git에 올라가지 않음)
- OUT_DIR : 학습 결과물 저장 위치 (기본: 이 레포의 outputs/). 완성되면 item-dimension-api/app/data/로 복사
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))   # core 패키지 import용

DATA_DIR = Path(os.getenv("DATA_DIR", REPO / "data"))
OUT_DIR = Path(os.getenv("OUT_DIR", REPO / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROOT = DATA_DIR / "Sample"
VS_ZIP = DATA_DIR / "Validation" / "01.원천데이터" / "VS.zip"
VL_ZIP = DATA_DIR / "Validation" / "02.라벨링데이터" / "VL.zip"
