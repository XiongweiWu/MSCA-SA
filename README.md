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

## Dataset

Please download MSCOCO from [Official Website](https://cocodataset.org/#download), and unzip it in ./data folder (./data/coco/). 


## MSCOCO Benchmark and model zoo

Model | Module | Backbone | AP@val |  AP@test-dev | Link
--- |:---:|:---:|:---:|:---:|:---:
CMRCN | MSSA-Adp  | R-50   |  38.2/43.0 |  | [Config+Model]() 
CMRCN | MSCA-Adp  | R-50   |  38.3/43.2 |  | [Config+Model]() 
CMRCN | MSSA      | R-50   |  38.3/43.3 | 38.8/43.5 | [Config+Model]() 
CMRCN | MSCA      | R-50   |  38.6/43.3 | 38.8/43.5 | [Config+Model]() 
CMRCN | MSSA-adp  | R-101  |  39.3/44.4 | 39.8/44.8 | [Config+Model]() 
CMRCN | MSCA-adp  | R-101  |  39.1/44.2 | 39.8/44.8 | [Config+Model]() 
CMRCN | MSSA-adp  | X-101  |   |  | [Config+Model]() 
CMRCN | MSCA-adp  | X-101  |  40.7/46.1  | 41.1/46.6 | [Config+Model]() 
MRCN | MSSA-adp   | R-50   |  36.7/40.1 | - | [Config+Model]() 
MRCN | MSCA-adp   | R-50   |  36.6/39.8 | - | [Config+Model]() 

[1] *We have re-trained some models and thus the results may be slightly different from the ones reported in the paper (~0.1%).* 

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

Example (test-dev):

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=${PORT:-991} tools/test.py  ./data/models/cmrcn_r101_mssa_adp/cmrcn_r101_mssa_adp.py  ./data/models/cmrcn_r101_mssa_adp/final.pth  --launcher pytorch  --format-only  --options "jsonfile_prefix=./cmrcn_r101_mssa_adp"
```