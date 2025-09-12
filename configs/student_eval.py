_base_ = [
    'mmseg::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmseg',
        'mmseg.datasets',
        'mmseg.models',
        'mmseg.engine',
        'mmseg.evaluation',
    'mmrazor.datasets.transforms',
    'mmrazor.engine.hooks.save_pred_mask_hook'
    ],
    allow_failed_imports=False
)

# 与训练保持一致的数据预处理
data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

# 学生模型用于评估/可视化
model = dict(
    _scope_='mmrazor',
    type='StudentHRNetW18',
    num_classes=1,
    norm_eval=True,
    data_preprocessor=data_preprocessor
)

data_root = 'tests/data/dataset3.1'
metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='mmrazor.CenterFitResize', size=(400, 400)),
    # nonzero 前景、invert=False 与教师评估一致
    dict(type='mmrazor.BinarizeSegLabel', mode='nonzero', invert=False),
    dict(type='PackSegInputs')
]

val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        pipeline=val_pipeline,
        metainfo=metainfo),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False
)

test_dataloader = val_dataloader

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 保存纯掩码PNG
custom_hooks = [
    dict(
        type='mmrazor.SavePredMaskHook',
        out_dir='work_dirs/vis_student',
        file_suffix='png',
        foreground_value=255
    )
]
