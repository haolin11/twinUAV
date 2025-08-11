from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


class SAMPredictor:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.available = False
        self.device = device or 'cpu'
        self.model = None
        try:
            if model_path is None:
                return
            # 尝试 segment-anything 官方实现
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
            self._sam_registry = sam_model_registry
            self._SamPredictor = SamPredictor
            # 猜测模型类型
            if 'vit_h' in model_path:
                model_type = 'vit_h'
            elif 'vit_l' in model_path:
                model_type = 'vit_l'
            else:
                model_type = 'vit_b'
            sam = sam_model_registry[model_type](checkpoint=model_path)
            if hasattr(sam, 'to'):
                sam = sam.to(self.device)
            self.model = self._SamPredictor(sam)
            self.available = True
        except Exception:
            self.available = False

    def segment(self, rgb_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """返回简单掩码列表，每项包含: {'mask': HxW bool, 'score': float}
        若不可用返回空列表。
        """
        if not self.available or self.model is None:
            return []
        try:
            rgb = rgb_bgr[..., ::-1].copy()
            self.model.set_image(rgb)
            # 全图网格点 (粗) 提示，降低计算量
            h, w = rgb.shape[:2]
            step = max(32, min(h, w)//10)
            points = []
            labels = []
            for y in range(step//2, h, step):
                for x in range(step//2, w, step):
                    points.append([x, y]); labels.append(1)
            if len(points) == 0:
                return []
            import numpy as np
            points = np.array(points)
            labels = np.array(labels)
            masks, scores, _ = self.model.predict(point_coords=points, point_labels=labels, multimask_output=True)
            results: List[Dict[str, Any]] = []
            for i in range(min(5, masks.shape[0])):
                results.append({'mask': masks[i].astype(bool), 'score': float(scores[i])})
            return results
        except Exception:
            return []


