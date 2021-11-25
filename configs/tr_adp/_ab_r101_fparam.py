_base_ = './_ab_r50_pafpn.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101), neck=dict(type='FPARAM', in_channels=[256, 512, 1024,2048]))

