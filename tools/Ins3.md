像素级 L2（feature）与 logits 蒸馏（KL/CWD）的区别

- logits 蒸馏：对比教师/学生的类别概率分布（logits→softmax/sigmoid），传递类别间“相对关系”（更像知识蒸馏的经典做法，常用 KL/CWD）。
- feature L2：对齐连续特征值（未归一化），更强调表示空间的一致性，和尺度敏感，数值稳定、简单直接。
- 二者可以叠加：logits 负责“任务输出”一致，feature 负责“表征对齐”，实践中组合通常更好。

“最后一层输出”和“logits 层”是不是一回事

- 取决于你的模型定义。
  - 若最后一层就是产生 logits 的分类/分割头（例如 final conv），feature L2 实际是在“logits 前”的特征上对齐；logits 蒸馏则在“概率分布”上对齐（softmax/sigmoid+温度）。
  - 若最后一层是 backbone 的一个卷积而非 decode_head，logits 蒸馏需要挂到真正的 head（例如 decode_head.conv_seg）。

“蒸馏点位要不要选中间层”

- 不必须，但强烈推荐至少再增加 1~2 个中间层（例如每个 stage 的输出）做特征蒸馏：
  - 早/中层：纹理、边缘、结构信息，对提升泛化有帮助；
  - 末层/logits：任务决策对齐。
- 注意通道/空间尺寸可能不同，需用 1x1 conv 或上/下采样对齐后再做 L2/CWD；各点位设较小权重，避免训练不稳。

如果你希望，我可以在不动你主体网络的情况下：

- 追加一两个中间 recorder（谨慎选择真实存在的模块名），加 1x1 投影与 L2；
- 或把现有 distill 点位改为 head logits + CWD，并保留像素 L2 作为辅助。













##### 

```

GitHub Copilot

- 输入格式：
  - 张量形状: [B, 3, 400, 400]
  - 类型/范围: float32，0~1（PIL→ToTensor 后），放在 CUDA 上
  - 来源: final_pil_image_filled → transforms.ToTensor() → unsqueeze(0) → .cuda()

- 输出格式（Mdecoder）：
  - 张量形状: [B, 1, 400, 400]
  - 含义: 水印区域的连续掩码图（概率/置信度图），不是离散坐标
  - 数值: 连续值（通常应视作概率/分数）。代码里直接用它做裁剪 utils.crop_mask_and_image，并在需要时用阈值0.5二值化
    - 例: combined_masks_binarized = where(combined_masks < 0.5, 0, 1)

- 结论：
  - Mdecoder 输出的是单通道的“掩码图”（连续值），不是坐标点位，也不是直接的0/1二值图。需要时再做 sigmoid/阈值得到0/1。
```

```
结构是 HRNet-V2-W48：通道为[48, 96, 192, 384]，stem(conv1/conv2) + layer1(Bottleneck) → transition1 → stage2(2分支) → transition2 → stage3(3分支，多个module) → transition3 → stage4(4分支)。
是的，stage4 只有1个 HighResolutionModule，包含4个分支（对应48/96/192/384），每个分支一般由若干个BasicBlock组成。
```

```
推理流程（400×400 ImageOps.fit、ToTensor[0–1]、单通道logits→sigmoid→概率灰度保存）
```

```
收到。这套工程关键信息我已对齐：400×400 ImageOps.fit→ToTensor[0–1]，HRnet 返回4尺度特征，cat 后经 last_layer 得到单通道 logits→sigmoid 存概率灰度，权重来自单个 pth，键前缀含 backbone.model.*。

```



















