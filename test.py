from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.video_desnow.data.folder_dataset import extract_number
from src.video_desnow.models.video_desnower import VideoDesnowerNet, VideoDesnowerNetConfig
from src.video_desnow.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--noisy-root", type=str, required=True)
    p.add_argument("--out-root", type=str, required=True)

    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def read_img(path: Path, image_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


def main() -> None:
    args = parse_args()
    noisy_root = Path(args.noisy_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model = VideoDesnowerNet(VideoDesnowerNetConfig(frames=args.frames, image_size=args.image_size)).to(args.device)
    load_checkpoint(args.ckpt, model=model, optimizer=None, map_location=args.device)
    model.eval()

    half = args.frames // 2
    videos = sorted(
        [p for p in noisy_root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )

    with torch.no_grad():
        for vid_dir in videos:
            files = sorted(list(vid_dir.iterdir()), key=lambda p: extract_number(p.name))
            if not files:
                continue

            out_dir = out_root / vid_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(len(files)), desc=f"video {vid_dir.name}", ncols=100):
                clip = []
                for j in range(i - half, i + half + 1):
                    jj = min(max(j, 0), len(files) - 1)
                    clip.append(to_tensor(read_img(files[jj], args.image_size)))
                x = torch.stack(clip, dim=0).unsqueeze(0).to(args.device)
                pred = model(x).squeeze(0).clamp(0, 1)
                pred_np = (pred.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                pred_np = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

                out_name = files[i].name.replace("input", "output")
                cv2.imwrite(str(out_dir / out_name), pred_np)


if __name__ == "__main__":
    main()
