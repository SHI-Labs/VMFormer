# Getting started with VMFormer

## Data Preparation

### Video Matting Foreground Dataset
* [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)

### Video Matting Background Dataset
* [BG20k](https://github.com/JizhiziLi/GFM)
* [DVM](https://github.com/nowsyn/DVM)

### Composited Testing Set
* [Google Drive](https://drive.google.com/file/d/1EPnwlVU1NSccubO1W1AuutyDad_0X2mo/view?usp=sharing)

## Training

We conducted all experiments on 8xA6000 GPUs. To train VMFormer with 8 GPUs, run:

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/mv3_vmformer.sh
```

## Inference & Evaluation

Evaluating on the composited testing set with 200 video clips:

```
CUDA_VISIBLE_DEVICES=0 python inference_vm.py --model_path path/to/model_weights --masks --num_frames 20 --img_path path/to/testing_sets
```


