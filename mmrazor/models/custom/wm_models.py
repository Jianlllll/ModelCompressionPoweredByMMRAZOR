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
class StudentHRNetW18WithSDecoderLoss(StudentHRNetW18):
    """Student wrapper with SDecoder-based payload supervision.

    This class decodes the student's segmentation logits together with the
    input image using WMdemo's SDecoder to predict a 100-bit payload and
    computes a BCE loss against ground-truth bits loaded by the model.

    Args:
        num_classes (int): Number of output channels for the student (default 1).
        norm_eval (bool): Freeze BN running stats during train.
        data_preprocessor: Inherited.
        payload_json (str): Path to labels.json mapping fileName -> 100-bit string.
        sdecoder_trainable (bool): Whether to train SDecoder. Default False (frozen).
        image_size (tuple[int,int]): Expected (H,W) for SDecoder. Default (400,400).
        message_length (int): Payload length. Default 100.
    """

    def __init__(self,
                 num_classes: int = 1,
                 norm_eval: bool = True,
                 data_preprocessor: Optional[Any] = None,
                 payload_json: Optional[str] = None,
                 sdecoder_trainable: bool = False,
                 image_size: tuple[int, int] = (400, 400),
                 message_length: int = 100,
                 sdecoder_ckpt: Optional[str] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(num_classes=num_classes,
                         norm_eval=norm_eval,
                         data_preprocessor=data_preprocessor)

        # Resolve project root to import WMdemo.model_maotai
        try:
            import sys, os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            wmdemo_dir = os.path.join(project_root, 'WMdemo')
            if wmdemo_dir not in sys.path:
                sys.path.insert(0, wmdemo_dir)
            from model_maotai import Decoder_Diffusion  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f'Failed to import WMdemo.model_maotai.Decoder_Diffusion: {e}')

        # Build SDecoder
        H, W = int(image_size[0]), int(image_size[1])
        self.sdecoder = Decoder_Diffusion(H, W, message_length)
        # Load weights if provided
        if sdecoder_ckpt is not None:
            import torch
            import os
            ckpt_path = os.path.abspath(sdecoder_ckpt)
            state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            self.sdecoder.load_state_dict(state, strict=False)
        if not sdecoder_trainable:
            for p in self.sdecoder.parameters():
                p.requires_grad_(False)
        self._sdecoder_trainable = bool(sdecoder_trainable)
        self._expected_size_hw = (H, W)
        self._payload_len = int(message_length)

        # Load payload mapping if provided; fallback to on-the-fly missing -> error
        import json
        self._name_to_bits: Optional[dict[str, list[int]]] = None
        if payload_json is not None:
            payload_path = os.path.abspath(payload_json)
            with open(payload_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Normalize to basename -> list[int]
            self._name_to_bits = {}
            for k, v in raw.items():
                name = os.path.basename(k)
                if isinstance(v, str):
                    bits = [1 if ch == '1' else 0 for ch in v.strip()]
                else:
                    bits = [int(x) for x in v]
                self._name_to_bits[name] = bits

    def _get_payload_from_samples(self, data_samples: Optional[List[BaseDataElement]]) -> torch.Tensor:
        if data_samples is None or len(data_samples) == 0:
            raise RuntimeError('data_samples are required to fetch gt_payload')
        # Prefer payload from mapping using filename to avoid pipeline coupling
        if self._name_to_bits is None:
            # Try to read directly from data_samples if provided by pipeline
            bits_list = []
            for ds in data_samples:
                payload = getattr(ds, 'gt_payload', None)
                if payload is None:
                    # Try metainfo key
                    meta = getattr(ds, 'metainfo', {})
                    payload = meta.get('gt_payload', None) if isinstance(meta, dict) else None
                if payload is None:
                    raise RuntimeError('gt_payload is not found in data_samples and no payload_json provided')
                bits = torch.as_tensor(payload, dtype=torch.float32)
                bits_list.append(bits)
            return torch.stack(bits_list, dim=0)

        # Use mapping by filename
        bits_list: list[torch.Tensor] = []
        import os
        for ds in data_samples:  # type: ignore[assignment]
            meta = getattr(ds, 'metainfo', {})
            filename = None
            if isinstance(meta, dict):
                filename = meta.get('ori_filename', None) or meta.get('filename', None)
            if not filename:
                # Fallback attribute
                filename = getattr(ds, 'img_path', None)
            if not filename:
                raise RuntimeError('Cannot resolve filename from data_samples for payload lookup')
            name = os.path.basename(str(filename))
            bits = self._name_to_bits.get(name)
            if bits is None:
                raise KeyError(f'Payload for image {name} not found in mapping')
            bits_tensor = torch.tensor(bits, dtype=torch.float32)
            bits_list.append(bits_tensor)
        return torch.stack(bits_list, dim=0)

    def loss(self, inputs: torch.Tensor,
             data_samples: Optional[List[BaseDataElement]] = None):
        import math
        import torch.nn.functional as F
        # Helper: rotate tensor batch by angle (degrees) around center, differentiable
        def rotate_tensor(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
            # img: [N,C,H,W]
            n, c, h, w = img.shape
            theta = img.new_zeros((n, 2, 3))
            rad = math.radians(angle_deg)
            cos, sin = math.cos(rad), math.sin(rad)
            theta[:, 0, 0] = cos
            theta[:, 0, 1] = -sin
            theta[:, 1, 0] = sin
            theta[:, 1, 1] = cos
            grid = F.affine_grid(theta, size=img.size(), align_corners=False)
            return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)

        # Ensure expected SDecoder size
        H, W = self._expected_size_hw
        image = inputs
        if image.shape[-2:] != (H, W):
            image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)

        # Import utils for ROI crop (kept identical to eval)
        try:
            import sys, os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            wmdemo_dir = os.path.join(project_root, 'WMdemo')
            if wmdemo_dir not in sys.path:
                sys.path.insert(0, wmdemo_dir)
            import utils as wmd_utils  # type: ignore
        except Exception as e:
            raise RuntimeError(f'Failed to import WMdemo.utils: {e}')

        # Per-angle decode with student's own mask (use raw logits for ROI thresholding)
        angles = [0.0, 3.0, -3.0]
        per_angle_probs: list[torch.Tensor] = []
        per_angle_bces: list[torch.Tensor] = []

        # Ground-truth bits
        gt_bits = self._get_payload_from_samples(data_samples).to(image.device)
        if gt_bits.dim() == 1:
            gt_bits = gt_bits.unsqueeze(0)

        for ang in angles:
            # Rotate input image
            img_rot = rotate_tensor(image, ang)
            # Forward student to get mask logits at this angle
            s_logit = self.backbone(img_rot)
            if s_logit.shape[-2:] != (H, W):
                s_logit = F.interpolate(s_logit, size=(H, W), mode='bilinear', align_corners=False)
            # ROI crop with raw logits (no sigmoid/no binarize)
            try:
                mask_crop, img_crop = wmd_utils.crop_mask_and_image(s_logit, img_rot)
            except Exception:
                mask_crop, img_crop = s_logit, img_rot
            # Decode probabilities via SDecoder
            y_prob = self.sdecoder(img_crop, mask_crop)
            # Ensure shape [N, payload_len]
            if y_prob.dim() == 1:
                y_prob = y_prob.unsqueeze(0)
            per_angle_probs.append(y_prob)
            # BCE to GT
            bce = F.binary_cross_entropy(y_prob, gt_bits, reduction='none')
            # reduce over bits, keep batch
            bce = bce.mean(dim=1)  # [N]
            per_angle_bces.append(bce)

        # Stack angles: probs[K][N,L] -> [K,N,L], bces[K][N] -> [K,N]
        probs_stack = torch.stack(per_angle_probs, dim=0)
        bces_stack = torch.stack(per_angle_bces, dim=0)
        # Select best angle per sample by min BCE
        best_idx = torch.argmin(bces_stack, dim=0)  # [N]
        # Gather best probs per sample
        K, N, L = probs_stack.shape[0], probs_stack.shape[1], probs_stack.shape[2]
        idx_expanded = best_idx.view(1, N, 1).expand(1, N, L)
        best_probs = probs_stack.gather(dim=0, index=idx_expanded).squeeze(0)  # [N,L]
        # Loss = mean BCE of best angle, weight 0.5
        loss_decode = F.binary_cross_entropy(best_probs, gt_bits, reduction='mean') * 6

        with torch.no_grad():
            acc_bits = (torch.round(best_probs) == gt_bits).float().mean()

        # Ensure distiller recorders see features from the official (padded) inputs
        # to avoid size mismatch with teacher features during L2 distillation.
        try:
            with torch.no_grad():
                _ = self.backbone(inputs)
        except Exception:
            pass

        return dict(loss_decode=loss_decode, acc_bits=acc_bits)

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
