#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

# Run the script
python3 main.py --arch resnet18 --batch-size 128 --workers 8 --custom-lr-multiplier cosine --log-iters 1 --lr .1 --dataset cifar --data /home/gridsan/stefanou/data --adv-train 0 --out-dir /home/gridsan/stefanou/cifar10/resnet18 --save-ckpt-iters 50
