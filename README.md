# Attention-based Feature Aggregation (ID: 9754)

## Introduction
CVPR-22 submission ID:9754


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation. 

## Environment used in the Paper

- Cuda: 11.1
- Pytorch: 1.9.0
- Torchvision: 0.10.0
- MMCV-full: 1.3.11

### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dataset

Please download MSCOCO from [Official Website](https://cocodataset.org/#download), and unzip it in ./data folder (./data/coco/). 


## MSCOCO Benchmark and model zoo

### Single-Scale Training with 12 epochs

Model | Module | Backbone | AP_M@val | AP_M@test | AP_B@val | AP_B@test | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
Deformable-DETR | -  | ResNet-50   |  - | - | 37.6 | 38.0 | -
CMRCN | Sum-Up  | ResNet-50   | 35.9  | 36.1 | 41.2 | 41.5 | -
CMRCN (ours) | MSSA-Adp  | ResNet-50   |  38.2 | 38.7 | 43.0 | 43.3 | [Config+Model](https://drive.google.com/drive/folders/1IsvKaSSoA_MzkqqZLJ1QvRSuUOSnM0u1?usp=sharing) 
CMRCN (ours)| MSCA-Adp  | ResNet-50   |  38.3 | 38.6 | 43.2 | 43.3 | [Config+Model](https://drive.google.com/drive/folders/1PfoFdVq4jJevW_PHaXY8J8QGSMw6HDt9?usp=sharing) 
CMRCN (ours)| MSSA      | ResNet-50   |  38.3 | 38.8 | 43.3 | 43.5 | [Config+Model](https://drive.google.com/drive/folders/1ZOWb2xfP1CvSo30GDyOa-yPUOfNFzj0f?usp=sharing) 
CMRCN (ours)| MSCA      | ResNet-50   |  38.6 | 38.8 | 43.3 | 43.5 | [Config+Model](https://drive.google.com/drive/folders/14DqzJ48Duo7LNYbUSp3gaLclnfIOIsmL?usp=sharing) 
CMRCN (ours)| MSSA-adp  | ResNet-101  |  39.3 | 39.8 | 44.4 | 44.8 | [Config+Model](https://drive.google.com/drive/folders/1uLE-Ykt0gzbxE3dTx4ciZZOQRLKR-XhH?usp=sharing) 
CMRCN (ours)| MSCA-adp  | ResNet-101  |  39.1 | 39.8 | 44.2 | 44.7 | [Config+Model](https://drive.google.com/drive/folders/18XDibJD1WZsIgguLWfLN6jeq78GSN6qg?usp=sharing)
CMRCN (ours)| MSSA-adp  | ResNeXt-101  | 40.7 | 41.2 | 46.3 | 46.7 | [Config+Model](https://drive.google.com/drive/folders/1WyiXPAL4w0DlegpY3bUshBun1cAePT5o?usp=sharing) 
CMRCN (ours)| MSCA-adp  | ResNeXt-101  | 40.7 | 41.1  | 46.1 | 46.6 | [Config+Model](https://drive.google.com/drive/folders/1P2bG83d-3nLmgoNPGj-wtMsme0q5JA0z?usp=sharing) 

[1] *We have re-trained some models and thus the results may be slightly different from the ones reported in the paper (~0.1%).*\
[2] *AP_M and AP_B denote the AP on Mask and Box evaluation. Test denotes MSCOCO test-dev 2017 set.*


### Multi-Scale Training with 36 epochs

Model | Module | Backbone | AP_M@val | AP_M@test | AP_B@val | AP_B@test | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
Deformable-DETR | -     | ResNet-50   |  -     |  -    | 44.5 | 44.9 | -
CMRCN | Sum-Up          | ResNet-50   |  38.5  |  38.7 | 44.3 | 44.5 | -
CMRCN (ours) | MSSA-Adp | ResNet-50   |  40.2  |  40.9 | 45.6 | 46.0 | [Config+Model](https://drive.google.com/drive/folders/1ZVnleimDeX4iLibhQQBxaxrQ7JBZAdlv?usp=sharing) 
CMRCN (ours)| MSCA-Adp  | ResNet-50   |  40.3  | 40.8 | 45.5 | 45.9  | [Config+Model](https://drive.google.com/drive/folders/1GVDF5OJ4rcc9VPx05IQIL0OAsGnhtR2f?usp=sharing) 

[1] *AP_M and AP_B denote the AP on Mask and Box evaluation. Test denotes MSCOCO test-dev 2017 set.*\
[2] *The Mem and FPS are for inference stage, where the models are evaluated on 8 A-100 cards with 10k images.*

## Train & Test

Train script:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-991}    tools/train.py  [config]  --launcher pytorch
```

Exmaple:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-991}    tools/train.py  ./data/models/cmrcn_r101_mssa_adp/cmrcn_r101_mssa_adp.py  --launcher pytorch
```

Test script:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=${PORT:-991} tools/test.py  [config]   [weights]  --launcher pytorch  --eval bbox segm
```

Example:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=${PORT:-991} tools/test.py  ./data/models/cmrcn_r101_mssa_adp/cmrcn_r101_mssa_adp.py  ./data/models/cmrcn_r101_mssa_adp/final.pth  --launcher pytorch  --eval bbox segm
```