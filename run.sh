#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} ./tools/train.py configs/tr_adp/mask_rcnn_r50_fpn_1x_coco_mssaadp.py  --launcher pytorch
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} ./tools/train.py configs/tr_adp/mask_rcnn_r50_fpn_1x_coco_mssaadpms.py  --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} ./tools/train.py configs/tr_adp/mask_rcnn_r50_fpn_1x_coco_mscaadpms.py  --launcher pytorch
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} ./tools/train.py configs/tr_adp/_ab_x101_cafpn.py   --launcher pytorch --resume-from work_dirs/_ab_x101_cafpn/epoch_10.pth
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} ./tools/train.py configs/tr_adp/_ab_r50_fparam.py   --launcher pytorch 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} ./tools/train.py configs/tr_adp/_ab_r101_fparam.py   --launcher pytorch 

