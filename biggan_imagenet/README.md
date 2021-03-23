# LeCam Regularization for training BigGAN on ImageNet

We provide the testing code that synthesizes images with the provided trained
model.

## Installation

Clone this repository
```
git clone https://github.com/google/lecam-gan.git
cd lecam-gan/biggan_imagenet
```

Install packages
```
conda create --name lcgan_tf python=3.6
conda activate lcgan_tf
pip install tensorflow==1.14
pip install -e .
```

## Testing

Synthesizing the images:
```
CUDA_VISIBLE_DEVICES=0 python generate.py --tfhub_url MODEL --classes [0, 5, 10]
```
You can specify the index of desired class with the `--class` command.

## Colab

You can use the Colab `LeCamGAN_Demo.ipynb` to load our model and generate the images.

## Notes
This repository is built based one the Implementation from [compare\_gan](https://github.com/google/compare_gan). Our models are trained using TPUs.
