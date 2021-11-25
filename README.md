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


## Benchmark and model zoo

Model | Module | Backbone | AP@val |  AP@test-dev | Link
--- |:---:|:---:|:---:|:---:|:---:
CMRCN | MSSA-Adp  | R-50   |   |  | [Model+Config]()
CMRCN | MSCA-Adp  | R-50   |   |  | [Model+Config]()
CMRCN | MSSA  | R-50   |   |  | [Model+Config]()
CMRCN | MSCA  | R-50   |   |  | [Model+Config]()
CMRCN | MSSA-adp  | R-101   |   |  | [Model+Config]()
CMRCN | MSCA-adp  | R-101   |   |  | [Model+Config]()
CMRCN | MSSA-adp  | X-101   |   |  | [Model+Config]()
CMRCN | MSCA-adp  | X-101   |   |  | [Model+Config]()
MRCN | MSSA-adp  | R-50   |   |  | [Model+Config]()
MRCN | MSCA-adp  | R-50   |   |  | [Model+Config]()

## Train & Test

Train script:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-991}    tools/train.py --config [config]  --work-dir [work-dir]  --launcher pytorch
```

Exmaple:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-991}    tools/train.py --config   --work-dir    --launcher pytorch
```

Test script:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=${PORT:-991} tools/test.py  [config]   [weights]  --launcher pytorch  --eval bbox segm
```

Example:

```
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=${PORT:-991} tools/test.py    --launcher pytorch  --eval bbox segm
```

 