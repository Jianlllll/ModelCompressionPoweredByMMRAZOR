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
待更新
```



评估：

```
待更新
```









#### 蒸馏核心配置文件：

文件信息：其中包含learning rate、Recorder蒸馏点位、batch size、max_epochs、patience、num_workers



| 文件、位置                   | 作用                  | 蒸馏方式            | 其他说明                       |
| ---------------------------- | --------------------- | ------------------- | ------------------------------ |
| mmrazor\configs\distill1.py  | HRNet-W18模型训练     | result distillation | 现有参数非训练时所用，仅作测试 |
| mmrazor\configs\distill2.py  | HRNet-W18模型**微调** | 同上                |                                |
| mmrazor\configs\distill4.py  | PIDNet-M模型训练      | 同上                |                                |
| mmrazor\configs\distill1.2py | HRNet-W18模型训练     | 解码损失+蒸馏损失   |                                |







#### FILES

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

keys Match版本:mmrazor\mmrazor\models\Mdecoder_210000.backbone.pth



##### dataset文件

| 文件                                | 说明                  | 其他说明 |
| ----------------------------------- | --------------------- | -------- |
| mmrazor\tests\data\dataset2         | 五百张无水印          | 调试用   |
| mmrazor\tests\data\dataset3_3k      | 3千张无水印           |          |
| mmrazor\tests\data\dataset4RemoveBG | 3千张带水印（无背景） |          |
| mmrazor\tests\data\xx               | 3千张带随机水印       |          |



#### 当前结果

| File‘sName                      | Model               | Epoch     |                      Dataset                      |                           训练方式                           | mIoU（学生/老师） | -5度~5度水印提取匹配率(min-max)% |
| ------------------------------- | :------------------ | --------- | :-----------------------------------------------: | :----------------------------------------------------------: | :---------------- | -------------------------------- |
|                                 | 教师模型 HRNet-W48  |           |                                                   | 同时优化编解码器和定位器,给定位网络加上解码损失。损失里包括了编码视觉损失，解码损失mse和定位损失bce |                   | 98-100                           |
| mmrazor\work_dirs\distill_52    | 学生模型1 HRNet-W18 | 52        |                   3000张无水印                    | L2loss(mse),<br>蒸馏点位： backbone.last_layer.3 的单通道输出 | 88.320/89.095     | 58-88                            |
| mmrazor\work_dirs\distill1_17   | 同上                | 17        |            3000张无水印加5000张有水印             |                             同上                             | 80.370/82.135     | 54-72                            |
| mmrazor\work_dirs\distill2_25   | 同上                | 52+**25** | 在学生模型1上使用”3000张无水印加5000张有水印“微调 |                             同上                             | 77.530/82.135     | 62-76                            |
| mmrazor\work_dirs\distill_4_18  | 同上                | **18**    |              **3000张有水印(修正)**               |                             同上                             | **87.37/87.82**   | **59-79**                        |
| mmrazor\work_dirs\distill2_4_16 | 同上                | **52+16** |      **在学生模型1上使用”3000有水印(修正)**       |                             同上                             | **58/87**         | **44-51**                        |
| mmrazor\work_dirs\distill4_4_7  | PIDNet_M            | 7         |              3000张**有**水印(修正)               |                             同上                             | 77.8/87           | 46-55                            |
| mmrazor\work_dirs\distill4_4_25 | 同上                | 25        |                 3000张**无水印**                  |                             同上                             | 81.6/88           | 44-61                            |
