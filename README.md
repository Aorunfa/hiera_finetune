# 说明
本项目在原hiera仓库的基础上，增加mae和下游任务训练代码
> 模型代码来源于[hiera](https://github.com/facebookresearch/hiera)
- [x] <input type="checkbox" disabled checked> 数据预处理
- [x] <input type="checkbox" disabled checked> mae训练
- [x] <input type="checkbox" disabled checked> hiera训练
- [x] <input type="checkbox" disabled > 单机多卡训练

# 快速开始
## 环境安装
```bash
git clone https://github.com/Aorunfa/hiera_finetune.git
conda create -n hiera python=3.10
conda activate hiera
cd ./hiera_finetune
pip install -r requirements.txt
```

## 数据准备
### * 微调数据集
使用kinetics-400的一个小微数据集快速验证训练过程，从[tiny-kinetics-400](https://github.com/Tramac/tiny-kinetics-400)下载数据放置于./dataset目录。
确保目录结构如下
```text
dataset/tiny-kinetics-400
├── annotations
│   ├── tiny_train.csv
│   └── tiny_val.csv
├── data_30fps_frames
│   ├── abseiling
│   │   ├── _4YTwq0-73Y_000044_000054
│   │   │   ├── frame_00001.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── air_drumming
│   │   ├── ...
│   └── ...
├──
```
### * MAE预训练数据集


## 单卡训练
### * 微调
修改`train_ft.py`的Config类文件，启动单卡训练脚本，训练过程指标将存储在`record/metric.csv`
```bash
python ./train_ft.py
```
### * 预训练MAE
修改`train_mae.py`的Config类文件，启动单卡训练脚本，训练过程指标将存储在`record/metric_mae.csv`
```bash
python ./train_mae.py
```

## 单机多卡训练
```bash
python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 ./train_ft_dist.py

python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 ./train_mae_dist.py
```

# 训练指标说明


