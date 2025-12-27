from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.video_desnow.data.folder_dataset import VideoFolderDataset, VideoFolderDatasetConfig
from src.video_desnow.losses import L1Loss
from src.video_desnow.models.video_desnower import VideoDesnowerNet, VideoDesnowerNetConfig
from src.video_desnow.utils.checkpoint import save_checkpoint
from src.video_desnow.utils.logger import build_logger
from src.video_desnow.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--noisy-root", type=str, required=True)
    p.add_argument("--gt-root", type=str, required=True)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.999)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=1)

    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--knn-mode", type=str, default="softmax", choices=["mean", "softmax"])
    p.add_argument("--knn-temperature", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--patch-size-kformer", type=int, default=4)
    p.add_argument("--kformer-depth", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger = build_logger(args.log_dir, name="train")
    logger.info(f"device={args.device}")

    ds = VideoFolderDataset(
        VideoFolderDatasetConfig(
            noisy_root=args.noisy_root,
            gt_root=args.gt_root,
            frames=args.frames,
            image_size=args.image_size,
        )
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    model = VideoDesnowerNet(
        VideoDesnowerNetConfig(
            frames=args.frames,
            image_size=args.image_size,
            channels=args.channels,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            k_max=args.k_max,
            k_min=args.k_min,
            knn_mode=args.knn_mode,
            knn_temperature=args.knn_temperature,
            dropout=args.dropout,
            patch_size_kformer=args.patch_size_kformer,
            kformer_depth=args.kformer_depth,
        )
    ).to(args.device)

    criterion = L1Loss()
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        pbar = tqdm(loader, ncols=100, desc=f"epoch {epoch}/{args.epochs}")
        for noisy_clip, target in pbar:
            noisy_clip = noisy_clip.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                pred = model(noisy_clip)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            step += 1

            if step % 50 == 0:
                pbar.set_postfix(l1=running / 50)
                running = 0.0

        logger.info(f"epoch={epoch} time={time.time() - t0:.1f}s")

        if epoch % args.save_every == 0:
            ckpt = Path(args.ckpt_dir) / "latest.pt"
            save_checkpoint(str(ckpt), model=model, optimizer=optim, epoch=epoch, step=step)
            logger.info(f"saved={ckpt}")

    logger.info("done")


if __name__ == "__main__":
    main()
