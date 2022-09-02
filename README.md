# 多模态情感分类

# How to use

- Step 1, download code source

```
git clone https://github.com/sailist/MMEmo
cd MMEmo
pip install -r requirements.txt
```

- Step 2, download dataset. (See [Dataset Download](#Dataset Download))
- Step 3, modify `./config.py`, change value of each dataset.
- Step 4, run command to reimplement. (See [Reimplement](#Reimplement))
- (Optional) Step 5, replace feature of your own. (See [Replace Features](#Replace Features))

# Dataset Download

The format of the `dataset` parameter `{dataset}-{feature_type}-[replaced feature]-{n_classes}`

like:

```
iemocap-cogmen-4 (raw features provided by cogmen, 4-way)
iemocap-cogmen-6 (raw features provided by cogmen, 6-way)
iemocap-cogmen-sbert-4 (raw features provided by cogmen, with text feature replaced by sbert feature, 4-way)
...
```

## IEMOCAP

- iemocap-cogmen-sbert-x

```
TODO
```

## MELD

## MOSEI




# Reimplement

## cogmen

- COGMEN COntextualized GNN based Multimodal Emotion
  recognitioN [paper](https://arxiv.org/abs/2205.02455) [code](https://github.com/exploration-lab/cogmen)

```
python3 train_mm.py --module=cogmen --dataset=iemocap-cogmen-sbert-4 --modality=atv --reimplement --device=0 
python3 train_mm.py --module=cogmen --dataset=iemocap-cogmen-sbert-6 --modality=atv --reimplement --device=0
```

## MMGCN

- MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in
  Conversation [paper](https://aclanthology.org/2021.acl-long.440.pdf) [code](https://github.com/hujingwen6666/MMGCN)

```
python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-sbert-4 --modality=atv --reimplement --device=0 
python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-sbert-6 --modality=atv --reimplement --device=0
```

## DAG-ERC

- Directed Acyclic Graph Network for Conversational Emotion
  Recognition [paper](https://arxiv.org/abs/2105.12907) [code](https://github.com/shenwzh3/DAG-ERC/)

```
python3 train_mm.py --module=dagerc --dataset=iemocap-cogmen-sbert-4 --modality=atv --reimplement --device=0 
python3 train_mm.py --module=dagerc --dataset=iemocap-cogmen-sbert-6 --modality=atv --reimplement --device=0
```

## DialogueGCN

- DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in
  Conversation [paper](https://arxiv.org/abs/1908.11540) [code](https://github.com/mianzhang/dialogue_gcn)

```
python3 train_mm.py --module=dgcn --dataset=iemocap-cogmen-sbert-4 --modality=atv --reimplement --device=0
python3 train_mm.py --module=dgcn --dataset=iemocap-cogmen-sbert-6 --modality=atv --reimplement --device=0
```

# Replace Features

## Text Feature

sentence tranformer feature(used in cogmen)

```
python3 preprocess_text.py
```

## Video Feature

### mmaction feature

> https://mmaction2.readthedocs.io/en/latest/recognition_models.html#tsn

```
tsn_r50_1x1x3_100e_kinetics400_rgb, configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth

tsn_r50_320p_1x1x8_100e_kinetics400_rgb, configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth

x3d_s_13x6x1_facebook_kinetics400_rgb, configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth

x3d_m_16x5x1_facebook_kinetics400_rgb, configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth

tsn_r50_1x1x16_50e_sthv2_rgb, configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20210816-5d23ac6e.pth
```

```
python3 preprocess_video.py --config=../mmaction/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth 
python3 preprocess_video.py --config=../mmaction/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth
python3 preprocess_video.py --config=../mmaction/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth --dataset=iemocap-cogmen-video-6
python3 preprocess_video.py --config=../mmaction/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth --dataset=iemocap-cogmen-video-6
```

# DIY

```
TODO
```


