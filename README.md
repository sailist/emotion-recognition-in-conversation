# 多模态情感分类

## 扫参

```
cd /home/admin/python/MMEmo ; bash scripts/cogmen.sh
cd /home/admin/python/MMEmo ; bash scripts/dagrec.sh
cd /home/admin/python/MMEmo ; bash scripts/mmgcn.sh
```

## mmaction checkpoints

> https://mmaction2.readthedocs.io/en/latest/recognition_models.html#tsn

```
tsn_r50_1x1x3_100e_kinetics400_rgb
configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth

tsn_r50_320p_1x1x8_100e_kinetics400_rgb
configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth

x3d_s_13x6x1_facebook_kinetics400_rgb
configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth

x3d_m_16x5x1_facebook_kinetics400_rgb
configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth

tsn_r50_1x1x16_50e_sthv2_rgb
https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20210816-5d23ac6e.pth

python3 demo/demo.py configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py ~/pretrain/mmaction/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth ~/video_demo/Ses01F_impro01.avi tools/data/kinetics/label_map_k400.txt
python3 demo/demo.py configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py ~/pretrain/mmaction/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth ~/video_demo/Ses01F_impro01.avi tools/data/kinetics/label_map_k400.txt
```

# 抽特征

```
python3 preprocess_video.py --config=../mmaction/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth
python3 preprocess_video.py --config=../mmaction/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth
python3 preprocess_video.py --config=../mmaction/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth --dataset=iemocap-cogmen-video-6
python3 preprocess_video.py --config=../mmaction/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py --checkpoint=~/pretrain/mmaction/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth --dataset=iemocap-cogmen-video-6

python3 preprocess_video.py --config=../mmaction/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py --checkpoint=~/pretrain/mmaction/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20210816-5d23ac6e.pth --suffix=_ss
python3 preprocess_video.py --config=../mmaction/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py --checkpoint=~/pretrain/mmaction/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20210816-5d23ac6e.pth --dataset=iemocap-cogmen-video-6 --suffix=_ss
```
