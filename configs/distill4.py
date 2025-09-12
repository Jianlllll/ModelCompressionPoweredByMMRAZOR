_base_ = [
    # 使用 mmseg 的默认 runtime，确保 default_scope=mmseg，从而使用 mmseg 的 DATASETS/PIPELINES
    'mmseg::_base_/default_runtime.py'
]

# 确保导入 mmseg 包，从而在注册表中注册其组件
custom_imports = dict(
    imports=[
        'mmseg',
        'mmseg.datasets',
        'mmseg.models',
        'mmseg.engine',
        'mmseg.evaluation',
    'mmseg.visualization',
    # 注册我们新增的标签二值化变换
    'mmrazor.datasets.transforms',
    # 注册自定义学生/教师包装器（包含 StudentPIDNet）
    'mmrazor.models.custom.wm_models'
    ],
    allow_failed_imports=False
)

# 按 epoch 打印日志头（例如 Epoch [1][10/..]），提升可读性
log_processor = dict(by_epoch=True)

# 教师网络预训练权重路径（使用已对齐为 backbone.* 的权重）
teacher_ckpt = 'mmrazor/models/Mdecoder_210000.backbone.pth'

# 学生微调起点权重（仅包含学生的 backbone.*/last_layer.* 键）
# 请将其设置为你提取好的学生权重文件，例如：
# 'work_dirs/distill_52/best_mIoU_epoch_52.student_backbone.pth'
student_ckpt = 'work_dirs/distill_52/best_mIoU_epoch_52.backbone.pth'

# 使用自定义已注册的教师/学生模型；你的任务为二值/单通道(num_classes=1)
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    # 分割数据预处理器（与 mmseg 对齐）
    data_preprocessor=dict(
        type='mmseg.SegDataPreProcessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    ),
    # 学生与教师均为单通道头（num_classes=1），与现有权重保持一致
    # 通过 init_cfg 从已有学生权重进行微调（而非从头训练）
    architecture=dict(
        type='StudentPIDNet',
        num_classes=1,
        norm_eval=True,
        # PIDNet-M 参数
        m=2, n=3, planes=64, ppm_planes=96, head_planes=128,
        augment=False,
        init_cfg=dict(type='Pretrained', checkpoint=student_ckpt)
    ),
    teacher=dict(type='TeacherHRNetW48', num_classes=1),
    teacher_ckpt=teacher_ckpt,
    # 先关闭学生的GT监督，专注蒸馏信号；后续可再开启
    calculate_student_loss=False,
    teacher_trainable=False,
    teacher_norm_eval=True,
    student_trainable=True,

    # 蒸馏器：像素级MSE
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            # 记录学生 PIDNet 的最终 1x1 卷积输出
            out=dict(type='ModuleOutputs', source='backbone.final_layer.conv2')
        ),
        teacher_recorders=dict(
            out=dict(type='ModuleOutputs', source='backbone.last_layer.3')
        ),
        # 将教师输出下采样到学生分辨率（教师≈1/4 → 学生≈1/8，比例 0.5）
        connectors=dict(
            down_t=dict(
                type='TorchFunctionalConnector',
                function_name='interpolate',
                func_args=dict(scale_factor=0.5, mode='bilinear', align_corners=False)
            )
        ),
        distill_losses=dict(
            loss_l2=dict(type='L2Loss', loss_weight=0.1, normalize=False, div_element=True)
        ),
        loss_forward_mappings=dict(
            loss_l2=dict(
                s_feature=dict(from_student=True, recorder='out'),
                # 教师特征经过下采样连接器对齐空间尺寸
                t_feature=dict(from_student=False, recorder='out', connector='down_t')
            ),
        )
    )
)

# 暂停验证：待模型 predict 返回分割结果后再启用评测
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU'])
find_unused_parameters = True

# 训练循环：显式设置为 x个 epoch；暂不进行验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)

# Checkpoint 保存：每 20 个 epoch 保存一次；保留最后一个
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=15, save_last=True, save_best='mIoU', rule='greater')
)

# 早停策略：连续 15 次验证（val_interval=1 即 15 个 epoch）mIoU 无提升则停止
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',   # 与 save_best 使用同一指标
        patience=10,
        rule='greater',
        min_delta=0.0
    )
]

# 优化器：降低初始学习率并启用梯度裁剪，避免小 batch 训练中数值爆炸
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

# 使用你提供的数据集（单通道掩码，文件名后缀为 _mask.png）
data_root = 'tests/data/dataset2'

metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='mmrazor.CenterFitResize', size=(400, 400)),
    # 将非零像素视为前景，零为背景
    dict(type='mmrazor.BinarizeSegLabel', mode='nonzero', to_dtype='uint8', invert=False),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='mmrazor.CenterFitResize', size=(400, 400)),
    dict(type='mmrazor.BinarizeSegLabel', mode='nonzero', to_dtype='uint8', invert=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
    img_suffix='.png',
    seg_map_suffix='_mask.png',
    pipeline=train_pipeline,
    metainfo=metainfo),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        img_suffix='.png',
        seg_map_suffix='_mask.png',
    pipeline=test_pipeline,
    metainfo=metainfo),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True
)

test_dataloader = None