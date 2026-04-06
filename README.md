# Divide-and-Conquer Predictor for Unbiased Scene Graph Generation
This repo is a CUDA 12-compatible unbiased SGG benchmark.

## Authors

**Xianjing Han**<sup>1</sup>, **Xingning Dong**<sup>1</sup>, **Xuemeng Song**<sup>1</sup>\*, **Tian Gan**<sup>1</sup>, **Yibing Zhan**<sup>2</sup>, **Yan Yan**<sup>3</sup>, **Liqiang Nie**<sup>1</sup>\*

<sup>1</sup> Shandong University  
<sup>2</sup> JD Explore Academy  
<sup>3</sup> Illinois Institute of Technology  
\* Corresponding author

## Recent Updates

- [04/2026] Benchmark and code upgraded for compatibility with recent CUDA 12 versions
- [06/2022] Initial release

## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Training on Scene Graph Generation](#Training-on-scene-graph-generation)
5. [Evaluation on Scene Graph Generation](#Evaluation)
6. [Citations](#Citations)

## Overview
This paper proposes a Divide-and-Conquer Predictor (DCNet) for scene graph generation that splits predicate prediction into a general pattern classification stage and multiple specific predicate classifiers, aiming to better distinguish visually similar relations.



## Training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols. 

For **Predicate Classification (PredCls)** on our DCNet:
``` bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR HierMotifsE2E SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /data/xianjing/others/vg  MODEL.PRETRAINED_DETECTOR_CKPT /data/xianjing/others/vg/pretrained_faster_rcnn/model_final.pth  OUTPUT_DIR /home/xianjing/sgg/checkpoints/DCNet-precls
```
For **Scene Graph Classification (SGCls)** on our DCNet:
``` bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR HierMotifsE2E SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /data/xianjing/others/vg  MODEL.PRETRAINED_DETECTOR_CKPT /data/xianjing/others/vg/pretrained_faster_rcnn/model_final.pth  OUTPUT_DIR /home/xianjing/sgg/checkpoints/DCNet-sgcls
```
For **Scene Graph Detection (SGDet)** on our DCNet:
``` bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR HierMotifsE2E SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /data/xianjing/others/vg  MODEL.PRETRAINED_DETECTOR_CKPT /data/xianjing/others/vg/pretrained_faster_rcnn/model_final.pth  OUTPUT_DIR /home/xianjing/sgg/checkpoints/DCNet-sgdet
```

## Evaluation

### Checkpoints
The cloud links of checkpoints: [Google Drive](https://drive.google.com/drive/folders/1kPU8DpKXsV_iMP8pce3JdadoDBSyeWJd?usp=sharing).

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Citations

``` bash
@article{han2022divide,
  title={Divide-and-conquer predictor for unbiased scene graph generation},
  author={Han, Xianjing and Dong, Xingning and Song, Xuemeng and Gan, Tian and Zhan, Yibing and Yan, Yan and Nie, Liqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={12},
  pages={8611--8622},
  year={2022},
  publisher={IEEE}
}
```
