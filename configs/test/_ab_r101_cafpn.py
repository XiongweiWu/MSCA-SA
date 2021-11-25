_base_ = './_ab_r50_cafpn.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

data = dict(test=dict(ann_file='data/coco/' + 'annotations/image_info_test-dev2017.json',
            img_prefix='data/coco/' + 'test2017/',
                     ))
