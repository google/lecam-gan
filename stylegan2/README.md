# LeCam Regularization for training StyleGAN2

Implementation of our regularization method for training the StyleGAN2 model
under the limited dataset.

## Installation

Clone this repository
```
git clone https://github.com/google/lecam-gan.git
cd lecam-gan/stylegan2
```

Install packages
```
conda create --name lcgan_tf python=3.6
conda activate lcgan_tf
pip install -r requirements.txt
```

### Preparation
Copy the file `lecam_loss.py` to the stylegan repository

Please refer to the instruction [here](https://github.com/NVlabs/stylegan2-ada) for processing the dataset.
Specifically, please use to the `num_samples` configuration in the
`dataset_tool.py` file to build limited CIFAR dataset.

## Training & Testing
Please refer to the scripts we provide in the `scripts` folder.

## Notes
This repository is built based one the Implementation from [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada).
