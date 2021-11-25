_base_ = [
    '../common/mstrain_3x_coco_instance.py',
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py'
]

model = dict(neck=dict(type='MSSAAdp', in_channels=[256, 512, 1024,2048]))

#optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
