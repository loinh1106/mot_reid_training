#!/bin/bash

python train.py -c 'configs/transformer.yaml' -o CHECKPOINT_PATH="" > logs/reid_transformer_$(date +%F-%H-%M-%S).log 2>&1