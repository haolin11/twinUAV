#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import numpy as np
import cv2 as cv
from .sam_segment import SAMPredictor


def estimate_depth_from_stereo(imgL: np.ndarray, imgR: np.ndarray, K: np.ndarray, baseline_m: float) -> np.ndarray:
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    sgbm = cv.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7, P1=8*3*7*7, P2=32*3*7*7, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    fx = float(K[0,0])
    with np.errstate(divide='ignore'):
        depth = fx * baseline_m / np.maximum(disp, 1e-6)
    depth[disp <= 0.0] = 0.0
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--sam_ckpt', type=str, default=None)
    ap.add_argument('--out', type=str, default='outputs/objects.json')
    args = ap.parse_args()

    data = np.load(args.dataset)
    imgsL = data['rgb_left']
    imgsR = data['rgb_right']
    K = data['K']
    predictor = SAMPredictor(args.sam_ckpt)
    results = []

    # 仅用首帧估计示例
    if len(imgsL) == 0:
        print('[Warn] dataset empty')
        print(json.dumps(results))
        return
    imgL = imgsL[0]; imgR = imgsR[0]
    depth = estimate_depth_from_stereo(imgL, imgR, K, baseline_m=0.05)
    masks = predictor.segment(imgL)
    H, W = imgL.shape[:2]

    for m in masks:
        mask = m['mask']
        if mask.shape != depth.shape:
            mask = cv.resize(mask.astype(np.uint8), (W, H), interpolation=cv.INTER_NEAREST).astype(bool)
        d = depth[mask]
        if d.size == 0:
            continue
        d_valid = d[(d > 0.05) & (d < 10.0)]
        if d_valid.size < 50:
            continue
        # 估计相机系物体中心
        ys, xs = np.where(mask)
        cx = float(xs.mean()); cy = float(ys.mean()); z = float(np.median(d_valid))
        fx, fy, px, py = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        X = (cx - px) * z / fx
        Y = (cy - py) * z / fy
        results.append({'center_cam': [X, Y, z], 'score': float(m.get('score', 0.0))})

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[OK] wrote {args.out}, num={len(results)}')


if __name__ == '__main__':
    main()


