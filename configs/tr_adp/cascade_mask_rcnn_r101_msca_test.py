_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco_mssa.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101), neck=dict(type='MSCAAdp', in_channels=[256, 512, 1024,2048]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)

data = dict(test=dict(ann_file='data/coco/' + 'annotations/image_info_test-dev2017.json',
            img_prefix='data/coco/' + 'test2017/',
                     ))
