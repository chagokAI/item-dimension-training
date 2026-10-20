"""물체 영역 기하 특징 + DINOv2 임베딩.

※ item-dimension-api의 app/measure.py와 같은 계산이어야 한다 (바꾸면 양쪽 모두 수정).
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from core import geometry

DINO = "facebook/dinov2-small"

Shape = tuple[np.ndarray, float]   # (윤곽 점들 (N, 2), 면적)


def shape_features(shape: Shape) -> list[float]:
    """shape 하나 -> 7개 특징 (bbox 가로/세로, 면적, 회전 사각형 긴/짧은 변, solidity, 종횡비)."""
    pts, area = shape
    _, _, w, h = geometry.bbox(pts)
    area = max(area, 1.0)
    hull = geometry.convex_hull(pts)
    long_, short = geometry.min_area_rect(hull)
    short = max(short, 1.0)
    solidity = area / max(geometry.polygon_area(hull), 1.0)
    return [np.log(max(w, 1.0)), np.log(max(h, 1.0)), np.log(area), np.log(long_), np.log(short), solidity, np.log(long_ / short)]


class Embedder:
    def __init__(self, device: torch.device):
        self.processor = AutoImageProcessor.from_pretrained(DINO)
        self.model = AutoModel.from_pretrained(DINO).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        e = self.model(**inputs).pooler_output.cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)
