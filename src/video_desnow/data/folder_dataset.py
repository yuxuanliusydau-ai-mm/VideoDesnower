from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_number(filename: str) -> int:
    parts = filename.split("_")
    number_part = parts[-1]
    number_part = number_part.replace("output", "")
    number_part = number_part.replace("input", "")
    return int(number_part.split(".")[0])


@dataclass(frozen=True)
class VideoFolderDatasetConfig:
    noisy_root: str
    gt_root: str
    frames: int = 5
    image_size: int = 224


class VideoFolderDataset(Dataset):
    def __init__(self, cfg: VideoFolderDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.noisy_root = Path(cfg.noisy_root)
        self.gt_root = Path(cfg.gt_root)
        self.half = cfg.frames // 2

        self.videos = sorted([p.name for p in self.noisy_root.iterdir() if p.is_dir()])
        if not self.videos:
            raise FileNotFoundError(f"No video folders found in {self.noisy_root}")

        self.index: List[Tuple[str, int]] = []
        self.frames_noisy: dict[str, List[Path]] = {}
        self.frames_gt: dict[str, List[Path]] = {}

        for vid in self.videos:
            noisy_files = sorted((self.noisy_root / vid).iterdir(), key=lambda p: extract_number(p.name))
            gt_files = sorted((self.gt_root / vid).iterdir(), key=lambda p: extract_number(p.name))
            if len(noisy_files) != len(gt_files):
                raise ValueError(f"Frame count mismatch in video {vid}: noisy={len(noisy_files)} gt={len(gt_files)}")

            self.frames_noisy[vid] = noisy_files
            self.frames_gt[vid] = gt_files

            n = len(noisy_files)
            for center in range(n):
                self.index.append((vid, center))

    def __len__(self) -> int:
        return len(self.index)

    def _read_img(self, path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(path))
        img = cv2.resize(img, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int):
        vid, center = self.index[idx]
        files_noisy = self.frames_noisy[vid]
        files_gt = self.frames_gt[vid]

        t = self.cfg.frames
        half = self.half

        clip = []
        for i in range(center - half, center + half + 1):
            j = min(max(i, 0), len(files_noisy) - 1)
            clip.append(self._read_img(files_noisy[j]))
        noisy_clip = torch.stack(clip, dim=0)

        target = self._read_img(files_gt[center])
        return noisy_clip, target
