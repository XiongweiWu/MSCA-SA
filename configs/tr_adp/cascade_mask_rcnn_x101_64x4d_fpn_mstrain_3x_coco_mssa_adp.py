_base_ = './cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_mssa_adp.py'

model = dict(
		pretrained='open-mmlab://resnext101_64x4d',
    	backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
	neck=dict(type='MSSAAdp', in_channels=[256, 512, 1024,2048]))


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
