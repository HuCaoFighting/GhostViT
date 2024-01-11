
# GhostViT: Expediting Vision Transformers via Cheap Operations

This repository is official implementaion for "GhostViT: Expediting Vision Transformers via Cheap Operations", which contains PyTorch evaluation code, training code and pretrained models.

For details see [GhostViT: Expediting Vision Transformers via Cheap Operations](https://ieeexplore.ieee.org/abstract/document/10292927) by Hu Cao, Zhongnan Qu, Guang Chen, Xinyi Li, Lothar Thiele, Alois Knoll.

If you use this code for a paper please cite:

```
@ARTICLE{10292927,
  author={Cao, Hu and Qu, Zhongnan and Chen, Guang and Li, Xinyi and Thiele, Lothar and Knoll, Alois},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={GhostViT: Expediting Vision Transformers Via Cheap Operations}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TAI.2023.3326795}}
```

# Model Zoo

We provide baseline GhostViT models pretrained on ImageNet 1k.

| name | acc@1  | url |
| --- | --- | --- | 
| GhostViT-tiny | 72.3 | [model](https://drive.google.com/drive/folders/1VOUu8_vd1-P1Hj7EB3J9Fh7J88wae9FD?usp=sharing) |
| GhostViT-small | 79.9 | [model](https://drive.google.com/drive/folders/1VOUu8_vd1-P1Hj7EB3J9Fh7J88wae9FD?usp=sharing) |

To load GhostViT with pretrained weights on ImageNet simply do:

```python
import torch
# check you have the right version of timm
import timm
assert timm.__version__ == "0.3.2"
```

# Usage

First, clone the repository locally:
```
git clone https://github.com/HuCaoFighting/GhostViT.git
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained GhostViT-small on ImageNet val with a single GPU run:

For GhostViT-small, run:
```
python main.py --eval --resume /path/to/model --model ghostFinalBestHeadMlp_deit_small_patch16_224 --data-path /path/to/imagenet
```

And for GhostViT-tiny:
```
python main.py --eval --resume /path/to/model  --model ghostFinalBestHeadMlp_deit_tiny_patch16_224 --data-path /path/to/imagenet
```

## Training
To train GhostViT-small and GhostViT-tiny on ImageNet-1k on a single node with 4 gpus for 300 epochs run:

GhostViT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ghostFinalBestHeadMlp_deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

GhostViT-tiny
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ghostFinalBestHeadMlp_deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## References
* [DeiT](https://github.com/facebookresearch/deit)
