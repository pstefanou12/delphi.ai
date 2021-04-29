#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

# Run the script
ipython3 main.py --arch vgg11 --batch-size 128 --workers 8 --custom-lr-multiplier cosine --log-iters 1 --lr .1 --dataset cifar --data data/ILSVRC adv-train 0 --out-dir  ../imagenet/vgg11 
