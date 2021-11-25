_base_ = './_ab_r50_pafpn.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101), neck=dict(type='PAFPN', in_channels=[256, 512, 1024,2048]))

data = dict(test=dict(ann_file='data/coco/' + 'annotations/image_info_test-dev2017.json',
            img_prefix='data/coco/' + 'test2017/',
                     ))
