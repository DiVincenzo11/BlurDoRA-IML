# BlurDoRA-IML

Core implementation for **Residual-Guided DoRA for Blur-Robust Image Manipulation Localization**.
This preliminary release makes the paper's main architectural choices inspectable.
It does not yet contain the full training or evaluation pipeline and cannot by itself
reproduce the reported scores.

## Current release

- `models/residual_branch.py`: Bayar-constrained 5×5 residual filter and four
  stride-2 Conv–BN–ReLU blocks, producing 1024 side tokens of width 256.
- `models/residual_injection.py`: four independent 256→487→1024 projection MLPs
  and zero-initialized scalar gates after DINOv3 blocks 6, 12, 18, and 24.
  Only patch tokens are changed; class and register tokens are preserved.
- `models/dora_config.py`: DoRA on each fused QKV projection with rank 128,
  alpha 256, and scale alpha/r = 2.
- `augmentations/blur.py`: one uniformly sampled Gaussian kernel from
  {3, 7, 11, 15, 19, 23} for every training image, before resizing.
- `configs/casia2_rg_dora.yaml`: key settings of the final CASIA2-only run.

The full training code, CAT-style data pipeline, evaluation scripts, additional
configurations, and checkpoints are planned for release upon publication.

## How the modules fit together

The method extends [DINOv3-IML](https://github.com/Irennnne/DINOv3-IML) and
uses the DINOv3 ViT-L/16 backbone. The modules here can be integrated as follows:

```python
from models.dora_config import attach_fused_qkv_dora
from models.residual_branch import ResidualBranch
from models.residual_injection import ResidualInjector

backbone = attach_fused_qkv_dora(backbone)  # DINOv3 ViT-L/16, 24 blocks
residual_branch = ResidualBranch()
injector = ResidualInjector()
injector.register(backbone)

# image: ImageNet-normalized [B, 3, 512, 512]
side_tokens = residual_branch(image)
with injector.use(side_tokens):
    features = backbone.get_intermediate_layers(
        image, n=1, reshape=True, norm=True
    )[0]

# Pass features to the DINOv3-IML localization head.
# Include residual_branch and injector parameters in the optimizer.
# During training, call this after every optimizer step:
residual_branch.bayar.project_constraints()
```

For the paper's blur schedule, place `six_level_blur()` from
`augmentations/blur.py` **before resizing and normalization** in the training
augmentation pipeline. Albumentations applies Gaussian blur to the RGB image
while leaving its mask unchanged. Evaluation masks must likewise remain sharp.

Pretrained DINOv3 weights, datasets, full paper checkpoints, and code from the
upstream repositories are not redistributed here. Obtain them under their
respective licenses. The architecture code is intended for inspection and
integration; the complete reproduction recipe will follow publication.

## Dependencies and attribution

Versions used to check this release are listed in `requirements.txt`. The model
builds on [DINOv3-IML](https://github.com/Irennnne/DINOv3-IML), whose upstream
code uses the MIT License. The DINOv3 model and benchmark datasets have their
own terms. No server paths, dataset manifests, training logs, or weights are
included in this repository.
