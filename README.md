### 环境配置

**Prerequisites**

虚拟环境搭建

```
conda create --name openmmlab python=3.10 -y
conda activate openmmlab


conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
(or pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116)
```



**Installation**

```
cd mmrazor
pip install -U openmim
python -m pip install -U joblib
mim install "mmengine>=0.7.0,<1.0.0"
mim install "mmcv==2.0.1"

python -m pip uninstall -y mmcls;
python -m pip install -i https://pypi.org/simple "mmcls==1.0.0rc6"
mim install mmsegmentation
pip install ftfy
pip install -v -e
pip install psutil
pip install kornia
pip install "opencv-python<4.10"
pip install "numpy==1.25.0"
pip install regex
```



参考：https://mmrazor.readthedocs.io/en/latest/get_started/installation.html



### Distillation

#### 模型选择

**学生模型选择**

| Model             | tensors | Parameters  | FLOPs | 参数内存占用(float32) |
| ----------------- | ------- | ----------- | ----- | --------------------- |
| 教师模型HRNet-W48 | 1839    | 66 M        | 120G  | 251.51 MB             |
| HRNet-W18         | 1839    | 9.7 M(-85%) | 24G   | 36.89 MB(-85%)        |
| Lite-HRNet        |         |             |       |                       |
| PIDNet-M          |         | 10 M(-85%)  |       |                       |



**模型来源** 

Lite-HRNet：https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py





#### 蒸馏训练PipeLine

1. 构建学生和教师模型文件
2. 将学生与教师模型文件(mmrazor\configs\model.py)都通过一个 BaseModel 包装类（mmrazor\mmrazor\models\custom\wm_models.py）在注册表中注册
3. distill.py (mmrazor\configs\distill.py)通过 type 名称引用
   - 如果有预训练权重，继续通过 init_cfg 加载



#### 指令与流程

蒸馏：

```
python tools/train.py configs/distillx.py
```



训练后

```
TBD
```



评估：

```
TBD
```









#### 蒸馏核心配置文件：

文件信息：其中包含learning rate、Recorder蒸馏点位、batch size、max_epochs、patience、num_workers



| 文件、位置                   | 作用                  | 蒸馏方式            | 其他说明                                                     |
| ---------------------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| mmrazor\configs\distill1.py  | HRNet-W18模型训练     | result distillation | 现有参数非训练时所用，仅作测试                               |
| mmrazor\configs\distill2.py  | HRNet-W18模型**微调** | 同上                |                                                              |
| mmrazor\configs\distill4.py  | PIDNet-M模型训练      | 同上                |                                                              |
| mmrazor\configs\distill1.2py | HRNet-W18模型训练     | 解码损失+蒸馏损失   | NOTE：<br/>由于训练的学生模型无法直接调用WMdemo\EMD_maotai.py.decoder2函数，所有加入了一个STUDENT_USE_TEACHER_MASK参数开关。这是一个调试验证学生教师两条pipeline中的处理效果是否一致的开关，当打开后学生pipeline线中的输出掩码会被教师Mdecoder的掩码输出取代。 |



**mmrazor\configs\distill1.2py原理解释：**

```
教师侧（ValEval）
直接调用 EMD_maotai.decoder2(img_path, Mdecoder, Sdecoder)，其内部执行：
ImageOps.fit(LANCZOS) → 0/±3° 旋转 + 黑边 EDT 填充 → 用 Mdecoder 原始输出做 ROI 裁剪 → SDecoder 解出 100 位概率 → 以 GT 的 BCE 选最优角。
最后对选中概率以阈值 0.5 取整计算 bit-acc，等价于 demo 的 get_secret_acc（逐位 round 后平均）。

学生侧（评测链路，Eval）
已与 decoder2 对齐：ImageOps.fit(LANCZOS) → 0/±3° 旋转（PIL 路径 + EDT 填边）→ 用学生“原始 logits”（注意：不是 sigmoid/二值化）+ utils.crop_mask_and_image 进行 ROI → SDecoder 输出概率 → 按 GT 的 BCE 选最优角 → 概率≥0.5 取整计算 bit-acc。
诊断开关：student_use_teacher_mask 可切换学生侧 ROI 的掩码来源为教师 Mdecoder。

学生侧（训练时解码损失，Train）
实现在 StudentHRNetW18WithSDecoderLoss.loss：
将输入插值到 400×400 → 用 grid_sample(padding_mode='border') 做 0/±3° 搜索（可导）→ ROI 仍用“原始 logits”+ crop_mask_and_image（在 CPU 上执行，再搬回 GPU）→ SDecoder 得到概率。
每个角与 GT 做 BCE，取最小 BCE 作为样本损失，batch 平均得到 loss_decode，再乘系数。
为避免与蒸馏 L2 的 Recorder 维度冲突，最后在 no_grad() 里用原始 inputs 再走一次 backbone，确保蒸馏特征来自 data_preprocessor 标准尺寸。

指标与钩子职责
bit_acc_metric.py（BitAccMetric）：产出 student_bit_acc/teacher_bit_acc，CheckpointHook 的 save_best 与 EarlyStoppingHook 的 monitor 均读取此指标，作为唯一真值来源。
bit_acc_eval_hook.py（BitAccEvalHook）：训练末批快速评估与验证日志/JSON 保存；
已移除对 MessageHub/metrics 的写入，避免与 evaluator 冲突；教师侧走 decoder2，学生侧走与 decoder2 一致的前处理与旋转搜索。

其他对齐细节
ImageOps.fit 使用 LANCZOS，旋转角度固定为 [0, +3, -3]，与 demo 一致。
黑边填充采用 EDT 逻辑（PIL→numpy→distance_transform_edt），与 demo 一致。
bit-acc 计算均以 0.5 阈值取整后逐位对比平均，顺序为“GT 在前，预测在后”，与 demo 一致。
SDecoder/Mdecoder 在评测中按 demo 习惯设为 train() 模式但包在 no_grad()，以复现行为（不影响梯度）。
```



#### 模型相关文件

##### 教师、学生模型定义文件：

| Model     | 文件位置                                                  | 其他                                                         |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| 教师模型  | mmrazor\configs\teacher_model.py                          |                                                              |
| HRNet-W18 | mmrazor\configs\student_model.py                          | channel widths：[18, 36, 72, 144]                            |
| PIDNet-M  | mmrazor\configs\model_utils.py；mmrazor\configs\pidnet.py | PIDNet-M: m=2, n=3, planes=64, head_planes=128, ppm_planes=96, augment=False, num_classes=1 |





##### 训练后学生模型权重文件：

mmrazor\work_dirs\



##### 教师模型权重文件:

原文件：mmrazor\mmrazor\models\Mdecoder_210000.pth





#### 数据集文件

##### dataset文件

| 文件                                | 说明                  | 其他说明 |
| ----------------------------------- | --------------------- | -------- |
| mmrazor\tests\data\dataset2         | 五百张无水印          | 调试用   |
| mmrazor\tests\data\dataset3_3k      | 3千张无水印           |          |
| mmrazor\tests\data\dataset4RemoveBG | 3千张带水印（无背景） |          |
| WMdemo/workdir/test\dataset6        | 3千张带随机水印       |          |
|                                     |                       |          |



#### 其他文件

| 文件                                                 | 说明                                                         | 其他说明 |
| ---------------------------------------------------- | ------------------------------------------------------------ | -------- |
| mmrazor\mmrazor\engine\hooks\bit_acc_eval_hook.py    | 训练/验证过程中的调试与记录用 Hook                           |          |
| mmrazor\mmrazor\evaluation\metrics\bit_acc_metric.py | 按与 decoder2一致的流程计算 student_bit_acc/teacher_bit_acc，并通过 evaluator 返回 metrics dict |          |
|                                                      |                                                              |          |
|                                                      |                                                              |          |



#### 当前结果

##### 总表：

| File‘sName                         | Model               | Epoch     |                        Dataset                        |                           训练方式                           | metric（学生/老师） | -5度~5度水印提取匹配率(min-max)% |
| ---------------------------------- | :------------------ | --------- | :---------------------------------------------------: | :----------------------------------------------------------: | :------------------ | -------------------------------- |
|                                    | 教师模型 HRNet-W48  |           |                                                       | 同时优化编解码器和定位器,给定位网络加上解码损失。损失里包括了编码视觉损失，解码损失mse和定位损失bce |                     | 98-100                           |
| mmrazor\work_dirs\distill_52       | 学生模型1 HRNet-W18 | 52        |                     3000张无水印                      | L2loss(mse),<br>蒸馏点位： backbone.last_layer.3 的单通道输出 | 88.320/89.095       | 58-88                            |
| mmrazor\work_dirs\distill1_17      | 同上                | 17        |            3000张无水印加5000张有固定水印             |                             同上                             | 80.370/82.135       | 54-72                            |
| mmrazor\work_dirs\distill2_25      | 同上                | 52+**25** | 在学生模型1上使用”3000张无水印加5000张有固定水印“微调 |                             同上                             | 77.530/82.135       | 62-76                            |
| mmrazor\work_dirs\distill_4_18     | 同上                | **18**    |              **3000张有固定水印(修正)**               |                             同上                             | **87.37/87.82**     | **59-79**                        |
| mmrazor\work_dirs\distill2_4_16    | 同上                | **52+16** |      **在学生模型1上使用”3000有固定水印(修正)**       |                             同上                             | **58/87**           | **44-51**                        |
| mmrazor\work_dirs\distill4_4_7     | PIDNet_M            | 7         |              3000张**有**固定水印(修正)               |                             同上                             | 77.8/87             | 46-55                            |
| mmrazor\work_dirs\distill4_4_25    | 同上                | 25        |                   3000张**无水印**                    |                             同上                             | 81.6/88             | 44-61                            |
| mmrazor\work_dirs\distill1.2_6_24e | HRNet-W18           | 24        |                    3000张随机水印                     |                  解码损失+掩码结果蒸馏损失                   | 88/94               | 83~100                           |



##### 重点结果筛选

| File‘sName                         | Model              | Epoch |    Dataset     |                           训练方式                           | metric（学生/老师） | -5度~5度水印提取匹配率(min-max)% |
| ---------------------------------- | :----------------- | ----- | :------------: | :----------------------------------------------------------: | :------------------ | -------------------------------- |
|                                    | 教师模型 HRNet-W48 |       |                | 同时优化编解码器和定位器,给定位网络加上解码损失。损失里包括了编码视觉损失，解码损失mse和定位损失bce |                     | 98-100                           |
| mmrazor\work_dirs\distill_52       | HRNet-W18          | 52    |  3000张无水印  |  L2loss(mse), 蒸馏点位： backbone.last_layer.3 的单通道输出  | 88.320/89.095       | 58-88                            |
| mmrazor\work_dirs\distill1.2_6_24e | HRNet-W18          | 24    | 3000张随机水印 |                  解码损失+掩码结果蒸馏损失                   | 88/94               | 83~100                           |
