_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='FCAT',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),)
