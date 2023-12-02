#!/bin/sh

cd fewshotbench

python run.py exp.name='prot_no_backbone_1_layers_0_contrastive_coef' method=protoformer dataset=swissprot_no_backbone lr=1e-6 +contrastive_coef=10 +protoformer_layer_num=1
python run.py exp.name='prot_no_backbone_1_layers_0_contrastive_coef' method=protoformer dataset=swissprot lr=1e-6 +contrastive_coef=0 +protoformer_layer_num=1 