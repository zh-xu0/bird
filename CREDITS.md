# Credits and Third-Party Notices

**BIRD — Bird Image Recognition Detector**  
Copyright (C) 2026 Zachary Xu and Jonathan Reppert.

The original source code in this repository is free software under the
**GNU General Public License v3.0 or later** (see [LICENSE](LICENSE)). Third-party
software, pretrained model weights, and image data remain under their own
terms, summarized below.

## Model and feature extractor

The classifier is a **frozen EfficientNet-B0** feature extractor followed by a
**StandardScaler + RBF SVM** trained on this project's dataset.

- **EfficientNet-B0** — Tan, M. and Le, Q. V. *EfficientNet: Rethinking Model
  Scaling for Convolutional Neural Networks.* ICML 2019.
  [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- **Implementation and ImageNet-1K pretrained weights** — `torchvision`
  (`EfficientNet_B0_Weights.IMAGENET1K_V1`, checkpoint
  `efficientnet_b0_rwightman-7f5810bc.pth`), part of the PyTorch project.
  torchvision is distributed under the **BSD-3-Clause** license, and the
  ImageNet-1K checkpoint is provided through its model zoo.
  <https://github.com/pytorch/vision>
- **PyTorch** — **BSD-3-Clause**.
  <https://github.com/pytorch/pytorch>

How it is used here: the pretrained classifier head is removed and the backbone
is evaluated in inference mode to produce a 1280-dimensional embedding per
image. The backbone is **not fine-tuned**; only the SVM is trained on the
project's images. The pretrained weights are downloaded by torchvision on
first use and are **not redistributed** in this repository.

- **scikit-learn** — `StandardScaler` and `SVC` (RBF kernel), **BSD-3-Clause**.
  Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12
  (2011), pp. 2825–2830. <https://scikit-learn.org>

The trained pipeline in `results/models/best_model.joblib` is part of this
project and is released under **GPL-3.0-or-later**.

## Dataset

The bird images come from [images.cv](https://images.cv). Full dataset:
7,340 training images and 310 test images across five classes. The full
dataset is not redistributed here; `game/demo_dataset/` contains a small demo
subset of 10 images per species.

- [Bald eagle](https://images.cv/dataset/bald-eagle-image-classification-dataset)
- [Bee-eater](https://images.cv/dataset/bee-eater-image-classification-dataset)
- [Bustard](https://images.cv/dataset/bustard-image-classification-dataset)
- [European gallinule](https://images.cv/dataset/european-gallinule-image-classification-dataset)
- [Sparrow](https://images.cv/search-labeled-image-dataset?query=sparrow)

Accessed 2026-11-21.

The images remain subject to **images.cv's terms of use** and the terms of
their original sources. They are **not** covered by this project's GPL-3.0
license, and this repository does not grant any rights to redistribute them
beyond what those terms allow. Dataset citations:

- Images.cv, *Computer Vision Datasets*, <https://images.cv>.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L.
  *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
- Russakovsky, O. et al. *ImageNet Large Scale Visual Recognition Challenge.*
  IJCV 115 (2015), pp. 211–252.

## Software dependencies

| Component | Version used | License |
|---|---|---|
| Python | 3.14.10 | PSF-2.0 |
| PyTorch (`torch`) | 2.15.0 | BSD-3-Clause |
| torchvision | 0.29.2 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| SciPy (scikit-learn dependency) | — | BSD-3-Clause |
| NumPy | 2.5.4 | BSD-3-Clause |
| pandas | 3.1.0 | BSD-3-Clause |
| Pillow | 13.0.0 | HPND (PIL Software License) |
| matplotlib | 3.11.2 | PSF-based matplotlib license |
| joblib | 1.6.0 | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| Tkinter / Tcl / Tk | bundled with Python | PSF / Tcl license |

Exact versions are pinned in [requirements.txt](requirements.txt). Full license
texts are included with each installed package and at the project links above.

No AGPL-licensed model or library (for example, Ultralytics YOLO) is used or
redistributed by this project.

## How to cite this project

```bibtex
@misc{xu2026bird,
  author = {Xu, Zachary and Reppert, Jonathan},
  title  = {{BIRD} --- Bird Image Recognition Detector},
  year   = {2026},
  note   = {Five-class bird image classifier and CAPTCHA-style game.
            GPL-3.0-or-later},
}
```

## Selected BibTeX entries

```bibtex
@inproceedings{tan2019efficientnet,
  author    = {Tan, Mingxing and Le, Quoc V.},
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year      = {2019},
  pages     = {6105--6114},
  url       = {https://arxiv.org/abs/1905.11946}
}

@inproceedings{pedregosa2011scikit,
  author    = {Pedregosa, Fabian and others},
  title     = {Scikit-learn: Machine Learning in Python},
  journal   = {Journal of Machine Learning Research},
  volume    = {12},
  pages     = {2825--2830},
  year      = {2011}
}

@inproceedings{deng2009imagenet,
  author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  title     = {ImageNet: A Large-Scale Hierarchical Image Database},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2009}
}
```
