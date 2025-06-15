# Marrying Autoregressive Transformer and Diffusion with Multi-Reference Autoregression <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.09482-b31b1b.svg)](https://arxiv.org/pdf/2506.09482)&nbsp;
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-TransDiff-yellow)](https://huggingface.co/zhendch/Transdiff)&nbsp;


<p align="center">
  <img src="figs/visual.png" width="720">
</p>

This is a PyTorch/GPU implementation of the paper [Marrying Autoregressive Transformer and Diffusion with Multi-Reference Autoregression](https://arxiv.org/pdf/2506.09482):

```
@article{zhen2025marrying,
  title={Marrying Autoregressive Transformer and Diffusion with Multi-Reference Autoregression},
  author={Zhen, Dingcheng and Qiao, Qian and Yu, Tan and Wu, Kangxi and Zhang, Ziwei and Liu, Siyuan and Yin, Shunshun and Tao, Ming},
  journal={arXiv preprint arXiv:2506.09482},
  year={2025}
}
```

This repo contains:

* 🪐 A simple PyTorch implementation of [TransDiff Model](models/transdiff.py) and [TransDiff Model with MRAR](models/transdiff_mrar.py)
* ⚡️ Pre-trained class-conditional TransDiff models trained on ImageNet 256x256 and 512x512
* 💥 A self-contained [notebook](demo.ipynb) for running various pre-trained TransDiff models
* 🛸 An TransDiff [training and evaluation script](main.py) using PyTorch DDP

## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### VAE Model
We adopt the VAE model from [MAR](https://github.com/LTH14/mar) , you can also get it [here](https://huggingface.co/zhendch/Transdiff/resolve/main/vae/checkpoint-last.pth?download=true).
### Installation

Download the code:
```
git clone https://github.com/TransDiff/TransDiff
cd TransDiff
```

A suitable [conda](https://conda.io/) environment named `transdiff` can be created and activated with:

```
conda env create -f environment.yaml
conda activate transdiff
```

For convenience, our pre-trained TransDiff models can be downloaded directly here as well:

| TransDiff Model                                                                                                                      | FID-50K | Inception Score | #params | 
|--------------------------------------------------------------------------------------------------------------------------------|---------|-----------------|---------|
| [TransDiff-B](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_b/checkpoint-last.pth?download=true)             | 2.47    | 244.2           | 290M    |
| [TransDiff-L](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_l/checkpoint-last.pth?download=true)             | 2.25    | 244.3           | 683M    |
| [TransDiff-H](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_h/checkpoint-last.pth?download=true)             | 1.69    | 282.0           | 1.3B    |
| [TransDiff-B MRAR](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_b_mrar/checkpoint-last.pth?download=true)   | 1.49    | 282.2           | 290M    |
| [TransDiff-L MRAR](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_l_mrar/checkpoint-last.pth?download=true)   | 1.61    | 293.4           | 683M    |
| [TransDiff-H MRAR](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_h_mrar/checkpoint-last.pth?download=true)   | 1.42    | 301.2           | 1.3B    |
| [TransDiff-L 512x512](https://huggingface.co/zhendch/Transdiff/resolve/main/transdiff_l_512/checkpoint-last.pth?download=true) | 2.51    | 286.6           | 683M    |

### (Optional) Download Other Files 
Download necessary [file](https://huggingface.co/zhendch/Transdiff/resolve/main/VIRTUAL_imagenet512.npz?download=true) and put it into folder `fid_stats/`, if you want to run evaluation on ImageNet 512x512.
Download [MRAR index file](https://huggingface.co/zhendch/Transdiff/resolve/main/Imagenet2012_mrar_files.txt?download=true) and put it into root of project, if you want to train TransDiff with MRAR.

### (Optional) Caching VAE Latents

Given that our data augmentation consists of simple center cropping and random flipping, 
the VAE latents can be pre-computed and saved to `CACHED_PATH` to save computations during TransDiff training:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```

## Usage

### Demo
Run our interactive visualization [demo](demo.ipynb).

### Training
Script for the TransDiff-L 1StepAR setting (Pretrain TransDiff-L with a width of 1024 channels, 800 epochs):
```
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model transdiff_large --diffloss_w 1024 \
--diffusion_batch_mul 4 \
--epochs 800 --warmup_epochs 100 --blr 1.0e-4 --batch_size 32 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```
- Training time is ~115h on 64 A100 GPUs with `--batch_size 32`.
- Add `--online_eval` to evaluate FID during training (every 50 epochs).
- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments. 
- (Optional) If the error 'Loss is nan, stopping training' frequently occurs during training when using mixed precision training with 'torch.cuda.amp.autocast()', you can add `--bf16` to the arguments.
- (Optional) If necessary, you can use gradient accumulation by setting `--gradient_accumulation_steps n`.

Script for the TransDiff-L MRAR setting (Finetune TransDiff-L MRAR with a width of 1024 channels, 40 epochs):
```
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model transdiff_large --diffloss_w 1024 --mrar --bf16 \
--diffusion_batch_mul 2 \
--epochs 40 --warmup_epochs 10 --lr 5.0e-5 --batch_size 16 --gradient_accumulation_steps 2 \
--output_dir ${OUTPUT_DIR} --resume ${Transdiff-L_1StepAR_DIR} \
--data_path ${IMAGENET_PATH}
```
Script for the TransDiff-L 512x512 setting (Finetune TransDiff-L 512x512 with a width of 1024 channels, 150 epochs):
```
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main.py \
--img_size 512 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model transdiff_large --diffloss_w 1024 --ema_rate 0.999 --bf16 \
--diffusion_batch_mul 4 \
--epochs 150 --warmup_epochs 10 --lr 1.0e-4 --batch_size 16 --gradient_accumulation_steps 2 \
--only_train_diff \
--output_dir ${OUTPUT_DIR} --resume ${Transdiff-L_1StepAR_DIR} \
--data_path ${IMAGENET_PATH}
```

### Evaluation (ImageNet 256x256 and 512x512)

Evaluate TransDiff-L 1StepAR with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model transdiff_large --diffloss_w 1024 \
--output_dir ${OUTPUT_DIR} --resume ckpt/transdiff_l/ \
--evaluate --eval_bsz 256 --num_images 50000 \
--cfg 1.3 --scale_0 0.89 --scale_1 0.95
```

Evaluate TransDiff-L MRAR with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model transdiff_large --diffloss_w 1024 \
--output_dir ${OUTPUT_DIR} --resume ckpt/transdiff_l_mrar/ \
--evaluate --eval_bsz 256 --num_images 50000 \
--cfg 1.3 --scale_0 0.91 --scale_1 0.93
```

Evaluate TransDiff-L 512x512 with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main.py \
--img_size 512 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model transdiff_large --diffloss_w 1024 \
--output_dir ${OUTPUT_DIR} --resume ckpt/transdiff_l_512/ \
--evaluate --eval_bsz 64 --num_images 50000 \
--cfg 1.3 --scale_0 0.87 --scale_1 0.87
```

More settings for Benchmark in paper:

| TransDiff Model     | cfg  | scale_0 | scale_1 | 
|---------------------|------|---------|---------|
| TransDiff-B         | 1.30 | 0.87    | 0.91    |
| TransDiff-L         | 1.30 | 0.89    | 0.95    |
| TransDiff-H         | 1.23 | 0.87    | 0.93    |
| TransDiff-B MRAR    | 1.30 | 0.87    | 0.91    |
| TransDiff-L MRAR    | 1.30 | 0.91    | 0.93    |
| TransDiff-H MRAR    | 1.28 | 0.87    | 0.91    |
| TransDiff-L 512x512 | 1.30 | 0.87    | 0.87    |

## Acknowledgements
A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar), [diffusers](https://github.com/huggingface/diffusers) and [timm](https://github.com/huggingface/pytorch-image-models).

## Contact

If you have any questions, feel free to contact me through email (zhendch@gmail.com). Enjoy!