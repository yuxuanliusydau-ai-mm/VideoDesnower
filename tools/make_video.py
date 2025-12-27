import argparse
import os

import cv2
import numpy as np


def extract_number(filename: str) -> int:
    parts = filename.split("_")
    number_part = parts[-1]
    number_part = number_part.replace("output", "")
    number_part = number_part.replace("input", "")
    return int(number_part.split(".")[0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=str, default="input_frames")
    p.add_argument("--output-root", type=str, default="output_frames")
    p.add_argument("--out-dir", type=str, default="video")
    p.add_argument("--num-videos", type=int, default=20)
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.num_videos):
        path = os.path.join(args.input_root, str(i))
        path2 = os.path.join(args.output_root, str(i))

        filelist = sorted(os.listdir(path), key=extract_number)
        predlist = sorted(os.listdir(path2), key=extract_number)

        size = (args.size * 2, args.size)
        out_path = os.path.join(args.out_dir, f"{i}.avi")
        video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc("I", "4", "2", "0"), args.fps, size)

        for img_name, pred_name in zip(filelist, predlist):
            img = cv2.imread(os.path.join(path, img_name))
            pred = cv2.imread(os.path.join(path2, pred_name))
            if img is None or pred is None:
                continue

            img = cv2.resize(img, (args.size, args.size))
            pred = cv2.resize(pred, (args.size, args.size))
            combined = np.concatenate([img, pred], axis=1)
            video.write(combined)

        video.release()


if __name__ == "__main__":
    main()
