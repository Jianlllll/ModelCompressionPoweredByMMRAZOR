from typing import Any, List, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement, PixelData
from mmrazor.registry import MODELS
from mmseg.structures import SegDataSample

# 直接从configs目录导入你现有的实现，避免搬迁大段代码
# 注意：这是相对工作目录的导入路径，训练时需从项目根目录启动
from configs.teacher_model import TeacherModel_HRNet_W48  # type: ignore
from configs.student_model import StudentModel_HRNet_W18  # type: ignore
from configs.pidnet import PIDNet  # type: ignore


class _BaseHRNetWrapper(BaseModel):
    """将普通nn.Module封装为BaseModel以适配MMRazor算法接口。

    - loss: 前向以产生中间特征供Recorder使用，返回空损失dict。
    - _forward: 返回张量，用于自定义张量前向。
    - predict: 返回空列表（此项目暂不使用指标评测）。
    """

    def __init__(self, net: torch.nn.Module, norm_eval: bool = False,
                 data_preprocessor: Optional[Any] = None) -> None:
        # 允许从配置传入 data_preprocessor（例如 mmseg.SegDataPreProcessor）
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = net
        # 不再将 last_layer 注册为顶层子模块，避免 state_dict 出现重复键
        self._norm_eval = norm_eval

    # 与BaseAlgorithm一致的签名
    def loss(self, inputs: torch.Tensor,
             data_samples: Optional[List[BaseDataElement]] = None):
        _ = self.backbone(inputs)
        return dict()

    def _forward(self, inputs: torch.Tensor,
                 data_samples: Optional[List[BaseDataElement]] = None):
        return self.backbone(inputs)

    def predict(self, inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None):
        logits = self.backbone(inputs)
        # 生成概率：
        # - 多通道：取前景通道的 softmax 概率（通道1）
        # - 单通道：对该通道做 sigmoid 概率
        if logits.dim() != 4:
            raise RuntimeError('Unexpected logits shape for segmentation prediction')
        if logits.shape[1] > 1:
            prob = torch.softmax(logits, dim=1)[:, 1:2, ...]
        else:
            prob = torch.sigmoid(logits)
        seg = (prob > 0.5).to(torch.long)

        preds: List[SegDataSample] = []
        bs = seg.shape[0]
        for i in range(bs):
            sample = SegDataSample()
            if data_samples is not None and i < len(data_samples):
                # 传递元信息（如图像形状）供 evaluator 使用
                sample.set_metainfo(getattr(data_samples[i], 'metainfo', {}))
                # 关键：将标签拷贝到输出样本，便于 IoUMetric 在验证时读取 gt
                # data_samples[i] 可能是 SegDataSample，包含 gt_sem_seg: PixelData
                if hasattr(data_samples[i], 'gt_sem_seg') and getattr(data_samples[i], 'gt_sem_seg') is not None:
                    sample.gt_sem_seg = data_samples[i].gt_sem_seg
            # 分割离散标签（0/1）
            sample.pred_sem_seg = PixelData(data=seg[i].detach().cpu())
            # 概率图（0-1）
            sample.pred_score = PixelData(data=prob[i].detach().cpu())  # type: ignore[attr-defined]
            preds.append(sample)
        return preds

    # 兼容无"backbone."前缀的checkpoint
    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]',
                        strict: bool = True):  # type: ignore[override]
        # 1) 兼容整包checkpoint：优先取 'model_state_dict' 或 'state_dict'
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict and isinstance(state_dict['model_state_dict'], dict):
                state_dict = state_dict['model_state_dict']  # type: ignore[assignment]
            elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
                state_dict = state_dict['state_dict']  # type: ignore[assignment]

        # 2) 去除常见外层前缀：DataParallel('module.'), 自定义('Mdecoder.')
        if isinstance(state_dict, dict):
            stripped = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                if new_k.startswith('Mdecoder.'):
                    new_k = new_k[len('Mdecoder.'):]
                stripped[new_k] = v
            state_dict = stripped  # type: ignore[assignment]

        # 3) 动态检测模型是否期望 'backbone.model.' 前缀
        model_keys_all = list(super().state_dict().keys())
        expect_backbone_model = any(k.startswith('backbone.model.') for k in model_keys_all)
        target_backbone_prefix = 'backbone.model.' if expect_backbone_model else 'backbone.'

        # 4) 统一映射为目标前缀（兼容 model./backbone.model./无前缀 等）
        if isinstance(state_dict, dict):
            normalized = {}
            for k, v in state_dict.items():
                key = k
                # 去掉已有的 backbone. 以便重组
                inner = key[len('backbone.'):] if key.startswith('backbone.') else key
                if target_backbone_prefix == 'backbone.model.' and not inner.startswith('model.'):
                    inner = 'model.' + inner
                if target_backbone_prefix == 'backbone.' and inner.startswith('model.'):
                    inner = inner[len('model.'):]
                new_k = 'backbone.' + inner
                normalized[new_k] = v
            state_dict = normalized  # type: ignore[assignment]

        # 5) 仅保留模型现有键
        model_keys = set(model_keys_all)
        filtered = {k: v for k, v in state_dict.items() if k in model_keys} if isinstance(state_dict, dict) else state_dict

        # 6) 非严格加载，减少刷屏
        result = super().load_state_dict(filtered, strict=False)
        print(f"[LoadSummary] matched={len(filtered)}/{len(model_keys)} target_prefix={target_backbone_prefix} ckpt_keys={len(state_dict) if isinstance(state_dict, dict) else 'N/A'}")
        return result

    # BaseModel 抽象 forward 的实现：与 BaseAlgorithm 一致的三态分派
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    # 训练模式时可选择将 BN 冻结，提升小 batch 稳定性
    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if mode and self._norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
                    # 关闭 BN 统计的更新
                    for p in m.parameters():
                        p.requires_grad_(False)
        return self


@MODELS.register_module()
class TeacherHRNetW48(_BaseHRNetWrapper):
    """教师模型包装为BaseModel，便于被算法构建和挂Recorder。"""

    def __init__(self, num_classes: int = 1, norm_eval: bool = False,
                 data_preprocessor: Optional[Any] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(TeacherModel_HRNet_W48(num_classes=num_classes),
                         norm_eval=norm_eval,
                         data_preprocessor=data_preprocessor)


@MODELS.register_module()
class StudentHRNetW18(_BaseHRNetWrapper):
    """学生模型包装为BaseModel，便于被算法构建和挂Recorder。"""

    def __init__(self, num_classes: int = 1, norm_eval: bool = False,
                 data_preprocessor: Optional[Any] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(StudentModel_HRNet_W18(num_classes=num_classes),
                         norm_eval=norm_eval,
                         data_preprocessor=data_preprocessor)


@MODELS.register_module()
class StudentPIDNet(_BaseHRNetWrapper):
    """PIDNet student wrapper.

    Defaults correspond to PIDNet-M (m=2, n=3, planes=64, head_planes=128).
    Set augment=False so forward returns a single logits tensor.
    """

    def __init__(self,
                 num_classes: int = 1,
                 norm_eval: bool = True,
                 data_preprocessor: Optional[Any] = None,
                 m: int = 2,
                 n: int = 3,
                 planes: int = 64,
                 ppm_planes: int = 96,
                 head_planes: int = 128,
                 augment: bool = False,
                 *args: Any, **kwargs: Any) -> None:
        net = PIDNet(m=m, n=n, num_classes=num_classes,
                     planes=planes, ppm_planes=ppm_planes,
                     head_planes=head_planes, augment=augment)
        super().__init__(net,
                         norm_eval=norm_eval,
                         data_preprocessor=data_preprocessor)

    def predict(self, inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None):
        logits = self.backbone(inputs)
        if logits.dim() != 4:
            raise RuntimeError('Unexpected logits shape for segmentation prediction')
        if logits.shape[1] > 1:
            prob = torch.softmax(logits, dim=1)[:, 1:2, ...]
        else:
            prob = torch.sigmoid(logits)

        # 仅对 PIDNet 学生在评估阶段进行上采样，以与标签尺寸一致
        target_h, target_w = int(inputs.shape[-2]), int(inputs.shape[-1])
        if data_samples is not None and len(data_samples) > 0:
            ds0 = data_samples[0]
            if hasattr(ds0, 'gt_sem_seg') and getattr(ds0, 'gt_sem_seg', None) is not None:
                gt_data = getattr(ds0.gt_sem_seg, 'data', None)
                if isinstance(gt_data, torch.Tensor) and gt_data.ndim >= 2:
                    target_h, target_w = int(gt_data.shape[-2]), int(gt_data.shape[-1])

        prob_up = F.interpolate(prob, size=(target_h, target_w), mode='bilinear', align_corners=False)
        seg = (prob_up > 0.5).to(torch.long)

        preds: List[SegDataSample] = []
        bs = seg.shape[0]
        for i in range(bs):
            sample = SegDataSample()
            if data_samples is not None and i < len(data_samples):
                sample.set_metainfo(getattr(data_samples[i], 'metainfo', {}))
                if hasattr(data_samples[i], 'gt_sem_seg') and getattr(data_samples[i], 'gt_sem_seg') is not None:
                    sample.gt_sem_seg = data_samples[i].gt_sem_seg
            sample.pred_sem_seg = PixelData(data=seg[i].detach().cpu())
            sample.pred_score = PixelData(data=prob_up[i].detach().cpu())  # type: ignore[attr-defined]
            preds.append(sample)
        return preds
