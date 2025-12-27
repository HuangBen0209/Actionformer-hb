# ActionFormer：基于 Transformer 的动作时刻定位

## 引言

本代码仓库实现了 ActionFormer，这是最早基于 Transformer 的时序动作定位模型之一 —— 用于检测动作实例的起始点和结束点，并识别其动作类别。在不借助额外复杂技巧的情况下，ActionFormer 在 THUMOS14 数据集上以 tIoU=0.5 的标准实现了 71.0% 的 mAP，比之前的最佳模型高出 14.1 个百分点，首次突破 60% mAP 大关。此外，ActionFormer 在 ActivityNet 1.3（平均 mAP 为 36.56%）和更具挑战性的 EPIC-Kitchens 100（比现有工作平均 mAP 高出 13.5%）上也展现出优异性能。我们的论文已被 ECCV 2022 接收，arXiv 版本可通过[此链接](https://arxiv.org/abs/2202.07925)查看。

此外，ActionFormer 是 2022 年 Ego4D Moment Queries 挑战赛中多个获胜方案的基础模型。我们的参赛作品尤其排名第二，创下 21.76% 的平均 mAP 和 42.54% 的 Recall@1x（tIoU=0.5）记录，几乎是官方基线的三倍。我们技术报告的 arXiv 版本可通过[此链接](https://arxiv.org/abs/2211.09074)查看。欢迎大家尝试使用本代码。



![](teaser.jpg)

具体而言，受 Transformer 在自然语言处理和计算机视觉领域近期成功的启发，我们采用极简设计，开发了一种基于 Transformer 的时序动作定位模型。如图所示，我们的方法采用局部自注意力机制来建模未裁剪视频中的时序上下文，对输入视频中的每个时刻进行分类，并回归其对应的动作边界。结果是一个可通过标准分类和回归损失进行训练的深度模型，能够在单次前向传播中定位动作时刻，无需使用动作提议或预定义的锚窗。

**相关项目**：

> [SnAG: 可扩展且准确的视频定位](https://arxiv.org/abs/2404.02257)
>
>  

穆方舟 \*、莫思成 \*、李尹&#x20;

*CVPR 2024*&#x20;



![github](https://img.shields.io/badge/-Github-black?logo=github)

&#x20;&#x20;



![github](https://img.shields.io/github/stars/fmu2/snag_release.svg?style=social)

&#x20;&#x20;



![arXiv](https://img.shields.io/badge/Arxiv-2404.02257-b31b1b.svg?logo=arXiv)

&#x20;

## 变更日志



* 2022 年 11 月 18 日：我们发布了用于[Ego4D Moment Queries（MQ）挑战赛](https://eval.ai/web/challenges/challenge-page/1626/overview)参赛作品的[技术报告](https://arxiv.org/abs/2211.09074)。代码仓库现已包含 Ego4D MQ 基准的配置文件、预训练模型和结果。

* 2022 年 8 月 29 日：更新了 arXiv 版本。

* 2022 年 8 月 1 日：更新代码仓库，包含 ActivityNet 上的最新结果。

* 2022 年 7 月 8 日：论文被 ECCV 2022 接收。

* 2022 年 5 月 9 日：更新了预训练模型。

* 2022 年 5 月 8 日：根据社区反馈和代码审查更新了代码仓库，使 THUMOS14 的平均 mAP 显著提升（>66.0%），ActivityNet 和 EPIC-Kitchens 100 的结果也略有改善。

## 代码概述

本代码仓库的结构深受 Detectron2 启发。主要组件包括：



* ./libs/core：参数配置模块。

* ./libs/datasets：数据加载和 IO 模块。

* ./libs/modeling：我们的主模型及其所有构建块。

* ./libs/utils：用于训练、推理和后处理的工具函数。

## 安装



* 请参考 INSTALL.md 安装必要的依赖并编译代码。

## 常见问题



* 参见 FAQ.md。

## 复现我们在 THUMOS14 上的结果

**下载特征和标注**



* 从[Box 链接](https://uwmadison.box.com/s/glpuxadymf3gd01m1cj6g5c3bn39qbgr)、[Google Drive 链接](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing)或[百度云链接](https://pan.baidu.com/s/1TgS91LVV-vzFTgIHl1AEGA?pwd=74eh)下载*thumos.tar.gz*（`md5sum 375f76ffbf7447af1035e694971ec9b2`）。

* 该文件包含 I3D 特征、json 格式的动作标注（与 ActivityNet 标注格式类似）以及外部分类分数。

**细节**：特征是从在 Kinetics 上预训练的双流 I3D 模型中提取的，使用`16帧`的片段，视频帧率为`~30 fps`，步长为`4帧`。这意味着每`4/30 ≈ 0.1333`秒对应一个特征向量。

**解压特征和标注**



* 将文件解压到 \*./data*目录下（或其他位置并链接到*./data\*）。

* 文件夹结构应如下所示：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───data/

│    └───thumos/

│    │         └───annotations

│    │         └───i3d\_features  &#x20;

│    └───...

|

└───libs

│

│   ...
```

**训练和评估**



* 使用 I3D 特征训练我们的 ActionFormer。这将在 \*./ckpt \* 目录下创建一个实验文件夹，用于存储训练配置、日志和检查点。



```
python ./train.py ./configs/thumos\_i3d.yaml --output reproduce
```



* \[可选] 使用 TensorBoard 监控训练过程



```
tensorboard --logdir=./ckpt/thumos\_i3d\_reproduce/logs
```



* 评估训练好的模型。预期的平均 mAP 应约为 62.6%，如我们主论文的表 1 所示。**在最近的提交中，预期平均 mAP 应高于 66.0%**。



```
python ./eval.py ./configs/thumos\_i3d.yaml ./ckpt/thumos\_i3d\_reproduce
```



* 在 THUMOS 上训练我们的模型需要约 4.5GB 的 GPU 内存，而推理可能需要超过 10GB 的 GPU 内存。建议使用至少 12GB 内存的 GPU。

**\[可选] 评估我们的预训练模型**

我们还提供了 THUMOS 14 的预训练模型。包含所有训练日志的模型可从[Google Drive 链接](https://drive.google.com/file/d/1isG3bc1dG5-llBRFCivJwz_7c_b0XDcY/view?usp=sharing)下载。要评估预训练模型，请按照以下步骤操作。



* 创建文件夹 \*./pretrained\*，并将文件解压到 \*./pretrained*目录下（或其他位置并链接到*./pretrained\*）。

* 文件夹结构应如下所示：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───pretrained/

│    └───thumos\_i3d\_reproduce/

│    │         └───thumos\_reproduce\_log.txt

│    │         └───thumos\_reproduce\_results.txt

│    │   └───...   &#x20;

│    └───...

|

└───libs

│

│   ...
```



* 训练配置记录在 \*./pretrained/thumos\_i3d\_reproduce/config.txt \* 中。

* 训练日志位于 \*./pretrained/thumos\_i3d\_reproduce/thumos\_reproduce\_log.txt*以及*./pretrained/thumos\_i3d\_reproduce/logs \* 中。

* 预训练模型为 \*./pretrained/thumos\_i3d\_reproduce/epoch\_034.pth.tar\*。

* 评估预训练模型。



```
python ./eval.py ./configs/thumos\_i3d.yaml ./pretrained/thumos\_i3d\_reproduce/
```



* 结果（不同 tIoU 下的 mAP）应如下表所示：



| 方法           | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 平均    |
| ------------ | ----- | ----- | ----- | ----- | ----- | ----- |
| ActionFormer | 82.13 | 77.80 | 70.95 | 59.40 | 43.87 | 66.83 |

## 复现我们在 ActivityNet 1.3 上的结果

**下载特征和标注**



* 从[Box 链接](https://uwmadison.box.com/s/aisdoymowukc99zoc7gpqegxbb4whikx)、[Google Drive 链接](https://drive.google.com/file/d/1VW8px1Nz9A17i0wMVUfxh6YsPCLVqL-S/view?usp=sharing)或[百度云链接](https://pan.baidu.com/s/1tw5W8B5YqDvfl-mrlWQvnQ?pwd=xuit)下载*anet\_1.3.tar.gz*（`md5sum c415f50120b9425ee1ede9ac3ce11203`）。

* 该文件包含 TSP 特征、json 格式的动作标注（与 ActivityNet 标注格式类似）以及外部分类分数。

**细节**：特征是从使用 TSP 在 ActivityNet 上预训练的 R (2+1) D-34 模型中提取的，使用`16帧`的片段，帧率为`15 fps`，步长为`16帧`（即**非重叠**片段）。这意味着每`16/15 ≈ 1.067`秒对应一个特征向量。特征已转换为 numpy 文件供我们的代码使用。

**解压特征和标注**



* 将文件解压到 \*./data*目录下（或其他位置并链接到*./data\*）。

* 文件夹结构应如下所示：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───data/

│    └───anet\_1.3/

│    │         └───annotations

│    │         └───tsp\_features  &#x20;

│    └───...

|

└───libs

│

│   ...
```

**训练和评估**



* 使用 TSP 特征训练我们的 ActionFormer。这将在 \*./ckpt \* 目录下创建一个实验文件夹，用于存储训练配置、日志和检查点。



```
python ./train.py ./configs/anet\_tsp.yaml --output reproduce
```



* \[可选] 使用 TensorBoard 监控训练过程



```
tensorboard --logdir=./ckpt/anet\_tsp\_reproduce/logs
```



* 评估训练好的模型。预期的平均 mAP 应约为 36.5%，如我们主论文的表 1 所示。



```
python ./eval.py ./configs/anet\_tsp.yaml ./ckpt/anet\_tsp\_reproduce
```



* 在 ActivityNet 上训练我们的模型需要约 4.6GB 的 GPU 内存，而推理可能需要超过 10GB 的 GPU 内存。建议使用至少 12GB 内存的 GPU。

**\[可选] 评估我们的预训练模型**

我们还提供了 ActivityNet 1.3 的预训练模型。包含所有训练日志的模型可从[Google Drive 链接](https://drive.google.com/file/d/1JKh3w14ngAjgzuuP22BnjhkhIcBSqteJ/view?usp=sharing)下载。要评估预训练模型，请按照以下步骤操作。



* 创建文件夹 \*./pretrained\*，并将文件解压到 \*./pretrained*目录下（或其他位置并链接到*./pretrained\*）。

* 文件夹结构应如下所示：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───pretrained/

│    └───anet\_tsp\_reproduce/

│    │         └───anet\_tsp\_reproduce\_log.txt

│    │         └───anet\_tsp\_reproduce\_results.txt

│    │   └───...   &#x20;

│    └───...

|

└───libs

│

│   ...
```



* 训练配置记录在 \*./pretrained/anet\_tsp\_reproduce/config.txt \* 中。

* 训练日志位于 \*./pretrained/anet\_tsp\_reproduce/anet\_tsp\_reproduce\_log.txt*以及*./pretrained/anet\_tsp\_reproduce/logs \* 中。

* 预训练模型为 \*./pretrained/anet\_tsp\_reproduce/epoch\_014.pth.tar\*。

* 评估预训练模型。



```
python ./eval.py ./configs/anet\_tsp.yaml ./pretrained/anet\_tsp\_reproduce/
```



* 结果（不同 tIoU 下的 mAP）应如下表所示：



| 方法           | 0.5   | 0.75  | 0.95 | 平均    |
| ------------ | ----- | ----- | ---- | ----- |
| ActionFormer | 54.67 | 37.81 | 8.36 | 36.56 |

**\[可选] 使用 I3D 特征复现我们的结果**



* 从[Google Drive 链接](https://drive.google.com/file/d/16239kUT2Z-j6S6PXIT1b_31OJi35QW_o/view?usp=sharing)下载*anet\_1.3\_i3d.tar.gz*（`md5sum e649425954e0123401650312dd0d56a7`）。

**细节**：特征是从在 Kinetics 上预训练的 I3D 模型中提取的，使用`16帧`的片段，帧率为`25 fps`，步长为`16帧`。这意味着每`16/25 = 0.64`秒对应一个特征向量。特征已转换为 numpy 文件供我们的代码使用。



* 将文件解压到 \*./data*目录下（或其他位置并链接到*./data\*），与 TSP 特征的存放方式类似。

* 使用 I3D 特征训练我们的 ActionFormer。这将在 \*./ckpt \* 目录下创建一个实验文件夹，用于存储训练配置、日志和检查点。



```
python ./train.py ./configs/anet\_i3d.yaml --output reproduce
```



* 评估训练好的模型。预期的平均 mAP 应约为 36.0%。这比我们论文中的结果略有提升，提升源于更好的训练方案 / 超参数（参见配置文件中的注释）。



```
python ./eval.py ./configs/anet\_i3d.yaml ./ckpt/anet\_i3d\_reproduce
```



* 包含所有训练日志的预训练模型可从[Google Drive 链接](https://drive.google.com/file/d/152dw2JDoNPssSnaQDaNolQUSFgcHlxe3/view?usp=sharing)下载。要生成结果，请创建文件夹 \*./pretrained\*，将文件解压到 \*./pretrained*目录下（或其他位置并链接到*./pretrained\*），然后运行：



```
python ./eval.py ./configs/anet\_i3d.yaml ./pretrained/anet\_i3d\_reproduce/
```



* 使用 I3D 特征的结果（不同 tIoU 下的 mAP）应如下表所示：



| 方法           | 0.5   | 0.75  | 0.95 | 平均    |
| ------------ | ----- | ----- | ---- | ----- |
| ActionFormer | 54.29 | 36.71 | 8.24 | 36.03 |

## 复现我们在 EPIC Kitchens 100 上的结果

**下载特征和标注**



* 从[Box 链接](https://uwmadison.box.com/s/vdha47qnce6jhqktz9g4mq1gc40w82yj)、[Google Drive 链接](https://drive.google.com/file/d/1Z4U_dLuu6_cV5NBIrSzsSDOOj2Uar85X/view?usp=sharing)或[百度云链接](https://pan.baidu.com/s/15tOdX6Yp4AJ9lFGjbQ8dgg?pwd=f3tx)下载*epic\_kitchens.tar.gz*（`md5sum add9803756afd9a023bc9a9c547e0229`）。

* 该文件包含 SlowFast 特征以及 json 格式的动作标注（与 ActivityNet 标注格式类似）。

**细节**：特征是从在 EPIC Kitchens 100 的训练集上预训练（动作分类）的 SlowFast 模型中提取的，使用`32帧`的片段，帧率为`30 fps`，步长为`16帧`。这意味着每`16/30 ≈ 0.5333`秒对应一个特征向量。

**解压特征和标注**



* 将文件解压到 \*./data*目录下（或其他位置并链接到*./data\*）。

* 文件夹结构应如下所示：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───data/

│    └───epic\_kitchens/

│    │         └───annotations

│    │         └───features  &#x20;

│    └───...

|

└───libs

│

│   ...
```

**训练和评估**



* 在 EPIC Kitchens 上，我们为名词和动词分别训练模型。

* 要使用 SlowFast 特征训练动词的 ActionFormer，请使用：



```
python ./train.py ./configs/epic\_slowfast\_verb.yaml --output reproduce
```



* 要使用 SlowFast 特征训练名词的 ActionFormer，请使用：



```
python ./train.py ./configs/epic\_slowfast\_noun.yaml --output reproduce
```



* 评估训练好的动词模型。预期的平均 mAP 应约为 23.4%，如我们主论文的表 2 所示。



```
python ./eval.py ./configs/epic\_slowfast\_verb.yaml ./ckpt/epic\_slowfast\_verb\_reproduce
```



* 评估训练好的名词模型。预期的平均 mAP 应约为 21.9%，如我们主论文的表 2 所示。



```
python ./eval.py ./configs/epic\_slowfast\_noun.yaml ./ckpt/epic\_slowfast\_noun\_reproduce
```



* 在 EPIC Kitchens 上训练我们的模型需要约 4.5GB 的 GPU 内存，而推理可能需要超过 10GB 的 GPU 内存。建议使用至少 12GB 内存的 GPU。

**\[可选] 评估我们的预训练模型**

我们还提供了 EPIC-Kitchens 100 的预训练模型。包含所有训练日志的动词模型可从[Google Drive 链接](https://drive.google.com/file/d/1Ta4ggKSj2YcszSrDbePlHe1ECF1CFKK4/view?usp=sharing)下载，名词模型可从[Google Drive 链接](https://drive.google.com/file/d/1OTlxeiWj8JE9n1-LsRYogHmqgUdsE5PR/view?usp=sharing)下载。要评估预训练模型，请按照以下步骤操作。



* 创建文件夹 \*./pretrained\*，并将文件解压到 \*./pretrained*目录下（或其他位置并链接到*./pretrained\*）。

* 文件夹结构示例如下：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───pretrained/

│    └───epic\_slowfast\_verb\_reproduce/

│    │         └───epic\_slowfast\_verb\_reproduce\_log.txt

│    │         └───epic\_slowfast\_verb\_reproduce\_results.txt

│    │   └───...  &#x20;

│    └───epic\_slowfast\_noun\_reproduce/

│    │         └───epic\_slowfast\_noun\_reproduce\_log.txt

│    │         └───epic\_slowfast\_noun\_reproduce\_results.txt

│    │   └───... &#x20;

│    └───...

|

└───libs

│

│   ...
```



* 训练配置记录在 \*./pretrained/epic\_slowfast\_(verb|noun)\_reproduce/config.txt \* 中。

* 训练日志位于 \*./pretrained/epic\_slowfast\_(verb|noun)*reproduce/epic\_slowfast*(verb|noun)*reproduce\_log.txt以及./pretrained/epic\_slowfast*(verb|noun)\_reproduce/logs \* 中。

* 预训练模型为 \*./pretrained/epic\_slowfast\_(verb|noun)*reproduce/epoch*(020|020).pth.tar\*。

* 评估预训练的动词模型。



```
python ./eval.py ./configs/epic\_slowfast\_verb.yaml ./pretrained/epic\_slowfast\_verb\_reproduce/
```



* 评估预训练的名词模型。



```
python ./eval.py ./configs/epic\_slowfast\_noun.yaml ./pretrained/epic\_slowfast\_noun\_reproduce/
```



* 结果（不同 tIoU 下的 mAP）应如下表所示：



| 方法               | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | 平均    |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| ActionFormer（动词） | 26.58 | 25.42 | 24.15 | 22.29 | 19.09 | 23.51 |
| ActionFormer（名词） | 25.21 | 24.11 | 22.66 | 20.47 | 16.97 | 21.88 |

## 复现我们在 Ego4D Moment Queries 基准上的结果

**下载特征和标注**



* 从[Ego4D 官网](https://ego4d-data.org/#download)下载官方的 SlowFast 和 Omnivore 特征，从[此链接](https://github.com/showlab/EgoVLP/issues/1#issuecomment-1219076014)下载官方的 EgoVLP 特征。请注意，我们无权发布这些特征和标注。相反，我们在`./tools/``convert_ego4d_trainval.py`提供了特征和标注转换脚本。

**细节**：所有特征均以`1.875 fps`的速率从`30 fps`的视频中提取。这意味着每`≈0.5333`秒对应一个特征向量。更多特征提取细节请参考 Ego4D 和 EgoVLP 的文档。

**解压特征和标注**



* 将文件解压到 \*./data*目录下（或其他位置并链接到*./data\*）。

* 文件夹结构应如下所示：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───data/

│    └───ego4d/

│    │   └───annotations

│    │   └───slowfast\_features

│    │   └───omnivore\_features

│    │   └───egovlp\_features &#x20;

│    └───...

|

└───libs

│

│   ...
```

**训练和评估**



* 我们提供了使用不同特征组合训练 ActionFormer 的配置文件。例如，使用 Omnivore 和 EgoVLP 特征训练将在 \*./ckpt \* 目录下创建一个实验文件夹，用于存储训练配置、日志和检查点。



```
python ./train.py ./configs/ego4d\_omnivore\_egovlp.yaml --output reproduce
```



* \[可选] 使用 TensorBoard 监控训练过程



```
tensorboard --logdir=./ckpt/ego4d\_omnivore\_egovlp\_reproduce/logs
```



* 评估训练好的模型。预期的平均 mAP 和 Recall@1x（tIoU=0.5）应分别约为 22.0% 和 40.0%。



```
python ./eval.py ./configs/ego4d\_omnivore\_egovlp.yaml ./ckpt/ego4d\_omnivore\_egovlp\_reproduce
```



* 使用所有三种特征在 Ego4D 上训练我们的模型需要约 4.5GB 的 GPU 内存，而推理可能需要超过 10GB 的 GPU 内存。建议使用至少 12GB 内存的 GPU。

**\[可选] 评估我们的预训练模型**

我们还提供了使用所有特征组合在 Ego4D 上训练的预训练模型。包含所有训练日志的模型可从[Google Drive 链接](https://drive.google.com/drive/folders/1NpAECS0ZhcCuehXkF9OhLQDPFrNdStJb?usp=sharing)下载。要评估预训练模型，请按照以下步骤操作。



* 创建文件夹 \*./pretrained\*，并将文件解压到 \*./pretrained*目录下（或其他位置并链接到*./pretrained\*）。

* 文件夹结构示例如下：



```
本文件夹

│   README.md

│   ... &#x20;

│

└───pretrained/

│    └───ego4d\_omnivore\_egovlp\_reproduce/

│    │   └───ego4d\_omnivore\_egovlp\_reproduce\_log.txt

│    │   └───ego4d\_omnivore\_egovlp\_reproduce\_results.txt

│    │   └───...  &#x20;

│    └───...

|

└───libs

│

│   ...
```



* 训练配置记录在 \*./pretrained/ego4d\_omnivore\_egovlp\_reproduce/config.txt \* 中。

* 训练日志位于 \*./pretrained/ego4d\_omnivore\_egovlp\_reproduce/ego4d\_omnivore\_egovlp\_reproduce\_log.txt*以及*./pretrained/ego4d\_omnivore\_egovlp\_reproduce/logs \* 中。

* 预训练模型为 \*./pretrained/ego4d\_omnivore\_egovlp\_reproduce/epoch\_010.pth.tar\*。

* 评估预训练模型。



```
python ./eval.py ./configs/ego4d\_omnivore\_egovlp.yaml ./pretrained/ego4d\_omnivore\_egovlp\_reproduce/
```



* 结果（不同 tIoU 下的 mAP）应如下表所示：



| 方法                  | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | 平均    |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| ActionFormer（S）     | 20.09 | 17.45 | 14.44 | 12.46 | 10.00 | 14.89 |
| ActionFormer（O）     | 23.87 | 20.78 | 18.39 | 15.33 | 12.65 | 18.20 |
| ActionFormer（E）     | 26.84 | 23.86 | 20.57 | 17.19 | 14.54 | 20.60 |
| ActionFormer（S+E）   | 27.98 | 24.46 | 21.21 | 18.56 | 15.60 | 21.56 |
| ActionFormer（O+E）   | 27.99 | 24.94 | 21.94 | 19.05 | 15.98 | 21.98 |
| ActionFormer（S+O+E） | 28.26 | 24.69 | 21.88 | 19.35 | 16.28 | 22.09 |



* 结果（不同 tIoU 下的 Recall@1x）应如下表所示：



| 方法                  | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | 平均    |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| ActionFormer（S）     | 52.25 | 45.84 | 40.60 | 36.58 | 31.33 | 41.32 |
| ActionFormer（O）     | 54.63 | 48.72 | 43.03 | 37.76 | 33.57 | 43.54 |
| ActionFormer（E）     | 59.53 | 54.39 | 48.97 | 42.75 | 37.12 | 48.55 |
| ActionFormer（S+E）   | 59.96 | 53.75 | 48.76 | 44.00 | 38.96 | 49.09 |
| ActionFormer（O+E）   | 61.03 | 54.15 | 49.79 | 45.17 | 39.88 | 49.99 |
| ActionFormer（S+O+E） | 60.85 | 54.16 | 49.60 | 45.12 | 39.87 | 49.92 |

## 在自定义数据集上训练和评估

工作进行中，敬请期待。

## 联系方式

李尹（yin.li@wisc.edu）

## 参考文献

如果您使用我们的代码，请考虑引用我们的论文。



```
@inproceedings{zhang2022actionformer,

&#x20; title={ActionFormer: Localizing Moments of Actions with Transformers},

&#x20; author={Zhang, Chen-Lin and Wu, Jianxin and Li, Yin},

&#x20; booktitle={European Conference on Computer Vision},

&#x20; series={LNCS},

&#x20; volume={13664},

&#x20; pages={492-510},

&#x20; year={2022}

}
```

如果您引用我们在 Ego4D 上的结果，除主论文外，请考虑引用我们的技术报告。



```
@article{mu2022actionformerego4d,

&#x20; title={Where a Strong Backbone Meets Strong Features -- ActionFormer for Ego4D Moment Queries Challenge},

&#x20; author={Mu, Fangzhou and Mo, Sicheng and Wang, Gillian, and Li, Yin},

&#x20; journal={arXiv e-prints},

&#x20; year={2022}

}
```

如果您使用 TSP 特征，请引用



```
@inproceedings{alwassel2021tsp,

&#x20; title={{TSP}: Temporally-sensitive pretraining of video encoders for localization tasks},

&#x20; author={Alwassel, Humam and Giancola, Silvio and Ghanem, Bernard},

&#x20; booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},

&#x20; pages={3173--3183},

&#x20; year={2021}

}
```

> （注：文档部分内容可能由 AI 生成）