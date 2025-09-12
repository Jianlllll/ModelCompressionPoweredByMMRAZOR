### Distillation

#### PipeLine

1. 构建学生和教师模型文件
2. 将学生与教师模型文件(mmrazor\configs\model.py)都通过一个 BaseModel 包装类（mmrazor\mmrazor\models\custom\wm_models.py）在注册表中注册
3. distill.py (mmrazor\configs\distill.py)通过 type 名称引用
   - 如果有预训练权重，继续通过 init_cfg 加载



#### 蒸馏核心超参文件：

文件信息：其中包含learning rate、Recorder蒸馏点位、batch size、max_epochs、patience、num_workers



| 文件、位置                  | 作用              | 蒸馏方式 | 其他说明 |
| --------------------------- | ----------------- | -------- | -------- |
| mmrazor\configs\distill1.py | HRNet-W18模型训练 |          |          |
| mmrazor\configs\distill2.py | HRNet-W18模型微调 |          |          |
| mmrazor\configs\distill4.py | PIDNet-M模型训练  |          |          |

准备使用argparse指令集成这几个核心蒸馏文件





#### 学生模型文件：

| Model     | 文件位置                                                  | 其他                                                         |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| HRNet-W18 | mmrazor\configs\student_model.py                          | channel widths：[18, 36, 72, 144]                            |
| PIDNet-M  | mmrazor\configs\model_utils.py；mmrazor\configs\pidnet.py | PIDNet-M: m=2, n=3, planes=64, head_planes=128, ppm_planes=96, augment=False, num_classes=1 |



#### 教师模型文件:

mmrazor\mmrazor\models\Mdecoder_210000.backbone.pth



#### dataset文件

| 文件                                | 说明                  | 其他说明 |
| ----------------------------------- | --------------------- | -------- |
| mmrazor\tests\data\dataset2         | 五百张无水印          | 调试用   |
| mmrazor\tests\data\dataset3_3k      | 3千张无水印           |          |
| mmrazor\tests\data\dataset4RemoveBG | 3千张带水印（无背景） |          |

