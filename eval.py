from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.video_desnow.data.folder_dataset import extract_number
from src.video_desnow.utils.metrics import psnr, ssim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", type=str, required=True)
    p.add_argument("--gt-root", type=str, required=True)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def read_img(path: Path, image_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)

    videos = sorted(
        [p for p in pred_root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )

    psnrs = []
    ssims = []

    for vid in videos:
        pred_files = sorted(list(vid.iterdir()), key=lambda p: extract_number(p.name))
        gt_dir = gt_root / vid.name
        gt_files = sorted(list(gt_dir.iterdir()), key=lambda p: extract_number(p.name))

        if len(pred_files) != len(gt_files):
            raise ValueError(f"Frame count mismatch in {vid.name}: pred={len(pred_files)} gt={len(gt_files)}")

        for pf, gf in tqdm(list(zip(pred_files, gt_files)), desc=f"video {vid.name}", ncols=100):
            pimg = read_img(pf, args.image_size)
            gimg = read_img(gf, args.image_size)
            psnrs.append(psnr(pimg, gimg, data_range=255.0))
            ssims.append(ssim(pimg, gimg, data_range=255.0))

    print(f"PSNR avg: {float(np.mean(psnrs)):.4f}")
    print(f"SSIM avg: {float(np.mean(ssims)):.6f}")


if __name__ == "__main__":
    main()
