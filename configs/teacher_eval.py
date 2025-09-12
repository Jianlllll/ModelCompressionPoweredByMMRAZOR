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

# 预处理与原脚本保持一致：不做均值方差标准化，不转换通道
data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255,255,255],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

# 仅构建教师模型用于评估
model = dict(
    _scope_='mmrazor',
    type='TeacherHRNetW48',
    num_classes=1,
    norm_eval=True,
    data_preprocessor=data_preprocessor
)

# 从 distill1 复用数据与评测配置（val集与二值化保持一致）

data_root = 'tests/data/dataset3.1'
metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='mmrazor.CenterFitResize', size=(400, 400)),
    # 一般二值掩码中白色(>0)代表前景，因此不取反；若你的数据相反，请改回 invert=True
    dict(type='mmrazor.BinarizeSegLabel', mode='nonzero', to_dtype='uint8', invert=False),
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

# 导出教师预测的纯掩码PNG
custom_hooks = [
    dict(
        type='mmrazor.SavePredMaskHook',
        out_dir='work_dirs/vis_teacher',
        file_suffix='png',
        foreground_value=255
    )
]
