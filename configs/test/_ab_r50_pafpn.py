_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(neck=dict(type='PAFPN', in_channels=[256, 512, 1024, 2048]))
data = dict(test=dict(ann_file='data/coco/' + 'annotations/image_info_test-dev2017.json',
                img_prefix='data/coco/' + 'test2017/',
                                     ))
