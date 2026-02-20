---
tasks:
- keyword-spotting
domain:
- audio
frameworks:
- pytorch
backbone:
- fsmn
metrics:
- Recall/FalseAlarm
license: Apache License 2.0
tags:
- Alibaba
- KWS
- WeKws
- FSMN
- CTC
- Mind Speech KWS
datasets:
  evaluation:
  - mobvoi_hotword_dataset_test
widgets:
  - task: keyword-spotting
    inputs:
      - type: audio 
        name: input 
        title: 音频 
    parameters:
      - type: string
        name: keywords
        title: 自定义唤醒词
    examples:
      - name: 1
        title: 示例1 
        inputs:
          - name: input
            data: git://example/nihaowenwen.wav
        parameters:
          - name: keywords
            value: 小白小白
    inferencespec:
      cpu: 1 #CPU数量
      memory: 1024 
---

# 语音唤醒模型介绍

## 模型描述

&emsp;&emsp;移动端语音唤醒模型，检测关键词为“你好问问”和“嗨小问”。  
&emsp;&emsp;模型网络结构继承自[论文](https://www.isca-speech.org/archive/interspeech_2018/chen18c_interspeech.html)《Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting》，其主体为4层cFSMN结构(如下图所示)，参数量约750K，适用于移动端设备运行。  
&emsp;&emsp;模型输入采用Fbank特征，训练阶段使用CTC-loss计算损失并更新参数，输出为基于char建模的中文全集token预测，token数共2599个。测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。  
&emsp;&emsp;模型训练采用"basetrain + finetune"的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用[出门问问开源数据](https://www.openslr.org/87/)进行微调得到输出模型。由于采用了中文char全量token建模，并使用充分数据进行basetrain，本模型支持基本的唤醒词/命令词自定义功能，但具体性能无法评估。如用户想验证更多命令词，可以通过页面右侧“在线体验”板块自定义设置并录音测试。  
&emsp;&emsp;目前最新ModelScope版本已支持用户在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型。欢迎您通过[小云小云](https://modelscope.cn/models/damo/speech_charctc_kws_phone-xiaoyun/summary)模型了解唤醒模型定制的方法。  

<p align="center">
<img src="fig/Illustration_of_cFSMN.png" alt="cFSMN网络框图" width="400" />
<p align="left">

## 使用方式和范围

运行范围：  
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。
- 模型训练需要用户服务器配置GPU卡，CPU训练暂不支持。

使用方式：
- 使用附带的kwsbp工具(Linux-x86_64)直接推理，分别测试正样本及负样本集合，综合选取最优工作点。

使用范围:
- 移动端设备，Android/iOS型号或版本不限，使用环境不限，采集音频为16K单通道。

目标场景:
- 移动端APP用到的关键词检测场景。

### 如何使用

请使用python3.9，并手动安装如下依赖：
```shell
pip install kwsbp==0.0.6 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

#### 模型推理代码范例：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

kwsbp_16k_pipline = pipeline(
    task=Tasks.keyword_spotting,
    model='damo/speech_charctc_kws_phone-wenwen')

kws_result = kwsbp_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/nihaowenwen.wav')
print(kws_result)
```

audio_in参数说明：
- 默认传入url地址的问问正样本音频，函数返回单条测试结果。
- 设置本地单条音频路径，如audio_in='LOCAL_PATH'，函数返回单条测试结果。
- 设置本地正样本目录(自动检索该目录下wav格式音频)，如audio_in=['POS_DIR', None]，函数返回全部正样本测试结果。
- 设置本地负样本目录(自动检索该目录下wav格式音频)，如audio_in=[None, 'NEG_DIR']，函数返回全部负样本测试结果。
- 同时设置本地正/负样本目录，如audio_in=['POS_DIR', 'NEG_DIR']，函数返回Det测试结果，用户可保存JSON格式文本方便选取合适工作点。

### 模型局限性以及可能的偏差

- 考虑到正负样本测试集覆盖场景不够全面，可能有特定场合/特定人群唤醒率偏低或误唤醒偏高问题。

## 训练数据介绍

- basetrain使用内部移动端ASR数据5000+小时，finetune使用[出门问问开源的关键词数据](https://www.openslr.org/87/)。这批数据包含了“你好问问”及“嗨小问”各2.5万条正样本，以及约220小时噪声数据，采集设备为远场麦克风（1/3/5米）。由于我们算法局限，问问的噪声数据无法标注不能直接用于训练，故采用ModelScope上开源的[离线ASR大模型](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)转写得到标注结果，并随机抽取2万条噪声数据用于训练。这种做法不太严谨，如条件允许应当将对应数据送标后使用。

## 模型训练流程

- 模型训练采用"basetrain + finetune"的模式，finetune过程使用目标场景的特定唤醒词数据并混合一定比例的负样本数据。如训练数据与应用场景不匹配，应当针对性做数据模拟。


### 预处理

- finetune模型直接使用出门问问开源数据，未做任何数据模拟。


## 数据评估及结果

- 模型在同一批开源的测试集上，FA选点为4小时一次，唤醒率：你好问问(98.46%)，嗨小问(97.64%)。

## 相关论文以及引用信息

```BibTeX
@inproceedings{chen18c_interspeech,
  author={Mengzhe Chen and ShiLiang Zhang and Ming Lei and Yong Liu and Haitao Yao and Jie Gao},
  title={{Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting}},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={2663--2667},
  doi={10.21437/Interspeech.2018-1204}
}
```

```BibTeX
@article{DBLP:journals/spl/HouSOHX19,
  author    = {Jingyong Hou and
               Yangyang Shi and
               Mari Ostendorf and
               Mei{-}Yuh Hwang and
               Lei Xie},
  title     = {Region Proposal Network Based Small-Footprint Keyword Spotting},
  journal   = {{IEEE} Signal Process. Lett.},
  volume    = {26},
  number    = {10},
  pages     = {1471--1475},
  year      = {2019},
  url       = {https://doi.org/10.1109/LSP.2019.2936282},
  doi       = {10.1109/LSP.2019.2936282}
}
```
