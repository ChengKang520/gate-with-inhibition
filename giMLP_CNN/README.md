
## giMLPs On CNNs On ImageNet Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

This is a PyTorch implementation of the paper [giMLPs: Gate with Inhibition Mechanism in MLPs]().

## Updates
- (05/08/2022) Initial release.



## Model Zoo

| Model                | Parameters | FLOPs    | Top 1 Acc. | Download |
| :------------------- | :--------- | :------- | :--------- | :------- |
| giCycleMLP-B1(0% Inhibition)          | 15M        |  2.1G    |  78.9%     |[model](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth)|
| giCycleMLP-B1(10% Inhibition)          | 15M        |  2.1G    |  78.8%     |[model](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth)|
| giCycleMLP-B1(30% Inhibition)          | 15M        |  2.1G    |  76.9%     |[model](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth)|

## Usage


### Install

- PyTorch 1.7.0+ and torchvision 0.8.1+
- [timm](https://github.com/rwightman/pytorch-image-models/tree/c2ba229d995c33aaaf20e00a5686b4dc857044be):
```
pip install 'git+https://github.com/rwightman/pytorch-image-models@c2ba229d995c33aaaf20e00a5686b4dc857044be'

or

git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models
git checkout c2ba229d995c33aaaf20e00a5686b4dc857044be
pip install -e .
```
- fvcore (optional, for FLOPs calculation)
- mmcv, mmdetection, mmsegmentation (optional)

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. You can run the batch file: ```sbatch imagenet_dowload.batch```

The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation
To evaluate a pre-trained giCycleMLP-B1 on ImageNet val with a single GPU run:
```
python main.py --eval --model giCycleMLP_B1 --resume path/to/giCycleMLP_B1.pth --data-path /path/to/imagenet
```


### Training

To train giCycleMLP-B1 with 30% inhibition level on ImageNet on a single node with 8 gpus for 300 epochs run:
```python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 --use_env main.py --model giCycleMLP_B1 --batch-size 128 --ratio_inhi 0.0 --dist_eval --data-path ./path/to/imagenet/ --output_dir ./B1/thre_00/```

```batch
#!/bin/sh
#SBATCH --partition=gpuextralong
#SBATCH --time=504:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=32000
#SBATCH --job-name=giMLP_B1_10
#SBATCH --err=giMLP_B1_10.err 
#SBATCH --out=giMLP_B1_10.out 

/bin/hostname
srun -l /bin/hostname
srun -l /bin/pwd
ml load Python/3.6.8-foss-2019a
cd /home/user/Rehrearsal_TransferLearning/python_script/giMLP_CNN/
source giMLPEnv/bin/activate
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 --use_env main.py --model CycleMLP_B1 --batch-size 128 --ratio_inhi 0.1 --dist_eval --data-path ./path/to/imagenet/ --output_dir ./B1/thre_10/
```

### Acknowledgement
This code is based on [CycleMLP](https://github.com/ShoufaChen/CycleMLP), [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). Thanks for their wonderful works

## License

giMLP is released under MIT License.
