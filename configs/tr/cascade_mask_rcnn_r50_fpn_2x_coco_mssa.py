_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)

model = dict(neck=dict(type='MSSA', in_channels=[256, 512, 1024, 2048]))

#lr_config = dict(step=[27, 33])
#runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

'''
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
'''
