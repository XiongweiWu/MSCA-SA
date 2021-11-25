_base_ = './_ab_r50_cafpn.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

