_base_ = './cascade_mask_rcnn_r50_fcat_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
