# LeCam Regularization for training BigGAN on CIFAR

Implementation of our regularization method for training the BigGAN model under the limited
CIFAR dataset.

## Installation

Clone this repository
```
git clone https://github.com/google/LeCamGAN.git
cd LeCamGAN/biggan_cifar
```

Install packages
```
conda create --name lcgan_pytorch python=3.6
conda activate lcgan_pytorch
conda install pytorch torchvision -c pytorch
pip install tensorflow==1.14
```

## Training
Please refer to the scripts we provide in the `scripts` folder. For example,
training the model on the 20% CIFAR-10 dataset with our regularization:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/lc-biggan-cifar10-0.2.sh
```

## Testing
Calculating the ID/FID scores with three evaluation runs:
```
CUDA_VISIBLE_DEVICES=0,1 python eval.py --repeat 3 --dataset C10 --network
weights/lc-biggan-cifar10-0.2/G_ema_best.pth
```
You can change the trained model file and dataset using the `--network` and
`--dataset` commands.

## Notes
This repository is built based one the Implementation from [DiffAug](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-biggan-cifar).
