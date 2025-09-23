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
    # 注册我们新增的标签二值化变换与hook
    'mmrazor.datasets.transforms',
    'mmrazor.engine.hooks'
    ],
    allow_failed_imports=False
)

# 按 epoch 打印日志头（例如 Epoch [1][10/..]），提升可读性
log_processor = dict(by_epoch=True)

# 教师网络预训练权重路径（使用已对齐为 backbone.* 的权重）
teacher_ckpt = 'mmrazor/models/Mdecoder_210000.backbone.pth'

# 数据与标签路径（相对当前工程启动目录 mmrazor）
DATA_ROOT = '../WMdemo/workdir/test'
TRAIN_IMG_DIR = 'train/img'
VAL_IMG_DIR = 'val/img'
TRAIN_PAYLOAD_JSON = DATA_ROOT + '/train/labels.json'
VAL_PAYLOAD_JSON = DATA_ROOT + '/val/labels.json'
SDECODER_CKPT = '../WMdemo/models/Sdecoder_210000.pth'
MDECODER_CKPT = '../WMdemo/models/Mdecoder_210000.pth'

# 调试开关：学生评测链路是否使用教师掩码
STUDENT_USE_TEACHER_MASK = False

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
    # 学生使用带SDecoder监督的封装
    architecture=dict(
        type='StudentHRNetW18WithSDecoderLoss',
        num_classes=1,
        norm_eval=True,
        payload_json=TRAIN_PAYLOAD_JSON,  # 训练阶段读取 train/labels.json
        sdecoder_trainable=False,
        sdecoder_ckpt=SDECODER_CKPT,
        image_size=(400, 400),
        message_length=100
    ),
    teacher=dict(type='TeacherHRNetW48', num_classes=1),
    teacher_ckpt=teacher_ckpt,
    # 开启学生损失（用于SDecoder BCE）
    calculate_student_loss=True,
    teacher_trainable=False,
    teacher_norm_eval=True,
    student_trainable=True,

    # 蒸馏器：像素级MSE，位点取最后输出卷积（last_layer.3）
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            out=dict(type='ModuleOutputs', source='backbone.last_layer.3')
        ),
        teacher_recorders=dict(
            out=dict(type='ModuleOutputs', source='backbone.last_layer.3')
        ),
        distill_losses=dict(
            loss_l2=dict(type='L2Loss', loss_weight=0.02, normalize=False, div_element=True)
        ),
        loss_forward_mappings=dict(
            loss_l2=dict(
                s_feature=dict(from_student=True, recorder='out'),
                t_feature=dict(from_student=False, recorder='out')
            ),
        )
    )
)

# 验证开启
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='Evaluator', metrics=[
    dict(type='BitAccMetric', image_size=(400, 400), mdecoder_ckpt=MDECODER_CKPT)
])
find_unused_parameters = True

# 训练循环：显式设置为 x个 epoch；每个 epoch 做一次 eval
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)


# Checkpoint 保存：以 student_bit_acc 作为最优
default_hooks = dict(
    #checkpoint=dict(type='CheckpointHook', interval=1, save_last=True, save_best='student_bit_acc', rule='greater')

    checkpoint=dict(
        type='CheckpointHook', 
        interval=10, 
        save_last=True, 
        save_best='student_bit_acc', 
        rule='greater'
    ),
    
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='student_bit_acc',  # The metric to monitor
        patience=15,               # Number of epochs to wait for improvement
        rule='greater'            # 'greater' means we want the metric to increase
    )

)

# 评测 Hook：计算学生/教师 bit-acc
custom_hooks = [
    dict(
        type='BitAccEvalHook',
        payload_json=VAL_PAYLOAD_JSON,
        image_size=(400, 400),
        mdecoder_ckpt=MDECODER_CKPT,
        train_payload_json=TRAIN_PAYLOAD_JSON,
        student_use_teacher_mask=STUDENT_USE_TEACHER_MASK,  # A/B 测试：学生端使用教师掩码
        # 确保在 CheckpointHook/EarlyStopping 之前写入 MessageHub 标量
        priority='VERY_HIGH'
    )
]

# 优化器：切换为 AdamW 并加入 5 个 epoch 的线性 warmup
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

# 线性 warmup + 余弦退火（按 epoch）
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=5, by_epoch=True, begin=5)
]

metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

# 训练使用 train/labels.json
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmrazor.LoadPayloadFromJSON', json_path=TRAIN_PAYLOAD_JSON),
    dict(type='PackSegInputs')
]

# 验证使用 val/labels.json
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmrazor.LoadPayloadFromJSON', json_path=VAL_PAYLOAD_JSON),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='BaseSegDataset',
        data_root=DATA_ROOT,
        data_prefix=dict(img_path=TRAIN_IMG_DIR),
        img_suffix='.png',
        pipeline=train_pipeline,
        metainfo=metainfo),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=False
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='BaseSegDataset',
        data_root=DATA_ROOT,
        data_prefix=dict(img_path=VAL_IMG_DIR),
        img_suffix='.png',
        pipeline=val_pipeline,
        metainfo=metainfo),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False
)

test_dataloader = None

# 覆盖默认 env_cfg：Windows 下使用 spawn
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)