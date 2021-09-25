dataset_type = 'CocoDataset'
data_root = '/kaggle/tmp/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625,
         scale_limit=0.1, rotate_limit=15, p=0.2),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.3),
    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
#    dict(
#        type="OneOf",
#        transforms=[
#            dict(type="Blur", p=1.0, blur_limit=7),
#            dict(type="GaussianBlur", p=1.0, blur_limit=7),
#            dict(type="MedianBlur", p=1.0, blur_limit=7),
#        ],
#        p=0.2,
#    ),
#albu_train_transforms = [
#    dict(
#        type='ShiftScaleRotate',
#        shift_limit=0.0625,
#        scale_limit=0.0,
#        rotate_limit=30,
#        interpolation=2,
#        p=0.5),
#    dict(
#        type='RandomBrightnessContrast',
#        brightness_limit=[0.1, 0.3],
#        contrast_limit=[0.1, 0.3],
#        p=0.2),
#    dict(
#        type='OneOf',
#        transforms=[
#            dict(
#                type='RGBShift',
#                r_shift_limit=10,
#                g_shift_limit=10,
#                b_shift_limit=10,
#                p=1.0),
#            dict(
#                type='HueSaturationValue',
#                hue_shift_limit=20,
#                sat_shift_limit=30,
#                val_shift_limit=20,
#                p=1.0)
#        ],
#        p=0.1),
#    dict(
#        type='OneOf',
#        transforms=[
#            dict(type='Blur', blur_limit=3, p=1.0),
#            dict(type='MedianBlur', blur_limit=3, p=1.0)
#        ],
#        p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='annotations/instances_train2017.json',
        img_prefix='images/train2017/',
        pipeline=train_pipeline,
        classes=('opacity',),
        data_root='/kaggle/tmp/coco/'),
    val=dict(
        type='CocoDataset',
        ann_file='annotations/instances_val2017.json',
        img_prefix='images/val2017/',
        pipeline=test_pipeline,
        classes=('opacity',),
        data_root='/kaggle/tmp/coco/'),
    test=dict(
        type='CocoDataset',
        ann_file='annotations/instances_test2017.json',
        img_prefix='images/test2017/',
        pipeline=test_pipeline,
        classes=('opacity',),
        data_root='/kaggle/tmp/coco/'))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=5e-5)
#e1: 0.005 - 1e-4, 
#e2: 0.001 - 5e-5, 
#e3: 5e-4 - 1e-5,
#e4: 5e-4 - 8e-6
#e5: 3e-4 - 5e-6
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=0.001,
#    step=[3, 5])
runner = dict(type='EpochBasedRunner', max_epochs=25)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/kaggle/input/mmdet-vfnet-pretrained/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1e-2),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
work_dir = './exps'
total_epochs = 25
seed = 42
gpu_ids = range(0, 1)
