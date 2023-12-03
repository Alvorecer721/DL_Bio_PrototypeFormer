#!/bin/sh

cd fewshotbench

python run.py exp.name='prot_no_backbone_1_layers_contrastive_loss_1_lr_1e-5' method=protoformer dataset=swissprot_no_backbone lr=1e-5 +contrastive_coef=1 +protoformer_layer_num=1
