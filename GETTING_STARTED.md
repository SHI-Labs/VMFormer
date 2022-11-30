# Getting started with VMFormer

## Data Preparation

### Foreground Dataset
* [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
* [ImageMatte](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md)

### Background Dataset
* [BG20k](https://github.com/JizhiziLi/GFM)
* [DVM](https://github.com/nowsyn/DVM)

### Composited Testing Set
* Low-resolution [Google Drive](https://drive.google.com/file/d/1xIsd9COmIRz4xr7olrdPmMeMWpvopd2h/view?usp=share_link)
* High-resolution [Google Drive](https://drive.google.com/file/d/1pkdZgAuSwepRzp2c5PYvj8nWmTOJy9EA/view?usp=share_link)

The way the data are composited following [RVM practice](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md).

## Training

We conducted all experiments on 8xA6000 GPUs. To train VMFormer with 8 GPUs, run:

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/mv3_vmformer.sh
```

## Inference & Evaluation

Evaluating VMFormer on the low-resoution composited testing set:

```
CUDA_VISIBLE_DEVICES=0 python inference_vm.py --model_path path/to/model_weights --masks --num_frames 20 --img_path path/to/vmformer_512x288_public --query_temporal weight_sum --fpn_temporal
```

Evaluating VMFormer on the high-resoution composited testing set:

```
CUDA_VISIBLE_DEVICES=0 python inference_vm.py --model_path path/to/model_weights --masks --num_frames 5 --img_path path/to/vmformer_1920x1080_public --query_temporal weight_sum --fpn_temporal
```

Evaluating VMFormer on the RVM low-resoution testing set:

```
CUDA_VISIBLE_DEVICES=0 python inference_rvm.py --model_path path/to/model_weights --masks --num_frames 20 --img_path path/to/rvm_512x288_public --query_temporal weight_sum --fpn_temporal
```

Evaluating VMFormer on the RVM high-resoution testing set:

```
CUDA_VISIBLE_DEVICES=0 python inference_rvm.py --model_path path/to/model_weights --masks --num_frames 20 --img_path path/to/rvm_1920x1080_public --query_temporal weight_sum --fpn_temporal
```
