#!/bin/sh

cd fewshotbench

python run.py exp.name='prot_no_backbone_1_layers_contrastive_loss_1_lr_3e-4' method=protoformer dataset=swissprot_no_backbone lr=3e-4 +contrastive_coef=1 +protoformer_layer_num=1
