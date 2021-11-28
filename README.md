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

Model | Module | Backbone | AP@val |  AP@test-dev | Link
--- |:---:|:---:|:---:|:---:|:---:
CMRCN | MSSA-Adp  | R-50   |  38.2/43.0 | 38.7/43.3 | [Config+Model](https://drive.google.com/drive/folders/1IsvKaSSoA_MzkqqZLJ1QvRSuUOSnM0u1?usp=sharing) 
CMRCN | MSCA-Adp  | R-50   |  38.3/43.2 | 38.6/43.3 | [Config+Model](https://drive.google.com/drive/folders/1PfoFdVq4jJevW_PHaXY8J8QGSMw6HDt9?usp=sharing) 
CMRCN | MSSA      | R-50   |  38.3/43.3 | 38.8/43.5 | [Config+Model](https://drive.google.com/drive/folders/1ZOWb2xfP1CvSo30GDyOa-yPUOfNFzj0f?usp=sharing) 
CMRCN | MSCA      | R-50   |  38.6/43.3 | 38.8/43.5 | [Config+Model](https://drive.google.com/drive/folders/14DqzJ48Duo7LNYbUSp3gaLclnfIOIsmL?usp=sharing) 
CMRCN | MSSA-adp  | R-101  |  39.3/44.4 | 39.8/44.8 | [Config+Model](https://drive.google.com/drive/folders/1uLE-Ykt0gzbxE3dTx4ciZZOQRLKR-XhH?usp=sharing) 
CMRCN | MSCA-adp  | R-101  |  39.1/44.2 | 39.8/44.7 | [Config+Model](https://drive.google.com/drive/folders/18XDibJD1WZsIgguLWfLN6jeq78GSN6qg?usp=sharing)
CMRCN | MSSA-adp  | X-101  |  40.7/46.3  | 41.2/46.7 | [Config+Model](https://drive.google.com/drive/folders/1WyiXPAL4w0DlegpY3bUshBun1cAePT5o?usp=sharing) 
CMRCN | MSCA-adp  | X-101  |  40.7/46.1  | 41.1/46.6 | [Config+Model](https://drive.google.com/drive/folders/1P2bG83d-3nLmgoNPGj-wtMsme0q5JA0z?usp=sharing) 
CMRCN (ms)| MSSA-Adp  | R-50   |  40.2/45.6 | 40.9/46.0 | [Config+Model](https://drive.google.com/drive/folders/1ZVnleimDeX4iLibhQQBxaxrQ7JBZAdlv?usp=sharing) 

[1] *We have re-trained some models and thus the results may be slightly different from the ones reported in the paper (~0.1%).* 
[2] *'ms' denotes that we train the models with multiscale input and longer training epochs (36 epochs).* 


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