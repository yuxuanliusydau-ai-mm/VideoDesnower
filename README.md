# Video Desnower: Adaptive Feature Fusion for Video Desnowing with Deformable Convolution and KNN Point Cloud Transformer


This repository reflects an early attempt in our exploration of image restoration.  After moving on to a new stage of research, the first author reorganized the relevant projects, and we share this version for reference.  We hope the released dataset and the auxiliary adaptive design (an extra weighting network for fusion) may provide some useful hints for related studies. For questions related to this repository, please use the following email address for correspondence:
yuxuan.li.usyd.au@gmail.com

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Datasets

### Download

Our dataset archive is available at:

```text
https://www.dropbox.com/scl/fi/ycu17t6gnbf650u292duj/dataset.zip?rlkey=z3wehcxfl9x2l6locezbeag75&st=kgweswfk&dl=0
```


### Expected folder structure

The loader assumes paired folders with identical video indices and frame counts:

```
INPUT_ROOT/
  0/
    *_input_0.png
    *_input_1.png
    ...
  1/
    ...
GT_ROOT/
  0/
    *_gt_0.png
    *_gt_1.png
    ...
```

Frames are sorted by the numeric suffix extracted from the filename.

## The main production process of our dataset:
1. **Collect clean videos** and extract frames as ground truth.
2. **Render a snow layer** for each frame by sampling snow particles with random spatial locations, sizes, and opacities. Particles can be rasterized as blurred disks, Gaussian blobs, or a combination of both.
3. **Generate streak-like patterns** by applying motion blur to the snow layer with randomly sampled directions and lengths. The blur parameters may vary across frames to mimic temporal dynamics, or remain fixed.
4. **Introduce photometric variations**, such as brightness jitter and slight color deviations around white, to reduce overly regular or uniform patterns.
5. **Composite** the snow layer onto the clean frames using alpha blending. Under heavier snow conditions, an additional mild veil or scattering term can be applied to approximate global accumulation effects.


---

## Training

```bash
python train.py   --noisy-root   --gt-root /path/to/GT_ROOT   --epochs    --batch-size    --frames    --image-size 
```


---

## Test (save predicted frames)

```bash
python test.py   --ckpt checkpoints/latest.pt   --noisy-root   --out-root output_frames   --frames    --image-size 
```

The script writes per-video folders into `output_frames/`.

---

## Evaluation (PSNR / SSIM)


```bash
python eval.py   --pred-root output_frames   --gt-root /path/to/GT_ROOT   --image-size 224
```

---

## Compose side-by-side videos (input vs output)

A script is provided to concatenate input and predicted frames horizontally and write AVI videos:

```bash
python tools/make_video.py   --input-root input_frames   --output-root output_frames   --out-dir video   --num-videos 20   --fps 100   --size 224
```

---

## Reference

If you use this code or dataset in academic work, please cite:

```bibtex
@ARTICLE{10606460,
  author={Li, Yuxuan and Dai, Lin},
  journal={IEEE Access}, 
  title={Video Desnower: An Adaptive Feature Fusion Understanding Video Desnowing Model With Deformable Convolution and KNN Point Cloud Transformer}, 
  year={2024},
  volume={12},
  number={},
  pages={104354-104366},
  keywords={Convolutional neural networks;Adaptation models;Nearest neighbor methods;Task analysis;Computational modeling;Deep learning;Computer vision;Videos;Snow;Computer vision;deep learning;video desnowing;feature fusion understanding},
  doi={10.1109/ACCESS.2024.3432709}
}
```

---

