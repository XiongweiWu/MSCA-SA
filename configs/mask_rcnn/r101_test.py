_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
data = dict(test=dict(ann_file='data/coco/' + 'annotations/image_info_test-dev2017.json',
                img_prefix='data/coco/' + 'test2017/',))
