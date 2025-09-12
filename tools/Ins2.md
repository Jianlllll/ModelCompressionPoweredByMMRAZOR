##### 指令

```
python -m pip install -U joblib
mim install "mmengine>=0.7.0,<1.0.0"
mim install "mmcv==2.0.1"

python -m pip uninstall -y mmcls;
python -m pip install -i https://pypi.org/simple "mmcls==1.0.0rc6"
mim install mmsegmentation
pip install ftfy

action:
python tools/train.py configs/distill1.py

python tools/train.py configs/distill2.py


eval:
python tools\test.py configs\teacher_eval.py mmrazor\models\Mdecoder_210000.pth


python tools\test.py configs\teacher_eval.py mmrazor\models\Mdecoder_210000.backbone.pth


(
python tools\changeToBackbone.py --input work_dirs\distill1_17\student_mapped_epoch_17.pth --output work_dirs\distill1_17\student_mapped_epoch_17.backbone.pth

to-mmrazor：统一成 backbone.*（你现在用的）
to-original：统一成原工程习惯的 backbone.model.，并把输出头移到 last_layer.
导出给原工程用的学生权重
python tmp.py --mode to-original --input work_dirs/distillXX/epoch_XX.pth --output work_dirs/distillXX/epoch_XX.for_wmdemo.pth
在原工程加载时，键名将是：
backbone.model.*（骨干）
last_layer.*（输出头）
)
python tools\test.py configs\student_eval.py work_dirs\distill1_17\student_mapped_epoch_17.backbone.pth



tools:
python tools\convert_ckpt.py mmrazor\models\Mdecoder_210000.pth mmrazor\models\Mdecoder_210000.conv.pth

python tools\inspect_hrnet_ckpt.py mmrazor\models\Mdecoder_210000.conv.pth

python tools\inspect_hrnet_ckpt.py mmrazor\models\Mdecoder_210000.pth

python tools\inspect_hrnet_ckpt.py work_dirs\distill_52\best_mIoU_epoch_52.pth
```











#####  





















##### 
