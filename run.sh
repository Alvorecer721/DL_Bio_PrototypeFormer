#!/bin/sh

cd fewshotbench

python run.py exp.name='prot_fcn_1_layers_1_contrastive_coef_lr_1e-3_l2_reg_0.1' method=protoformer dataset=swissprot lr=1e-3 +contrastive_coef=1 +protoformer_layer_num=1 weight_decay=0.1
python run.py exp.name='prot_fcn_1_layers_0_contrastive_coef_lr_1e-3_l2_reg_0.1' method=protoformer dataset=swissprot lr=1e-3 +contrastive_coef=0 +protoformer_layer_num=1 weight_decay=0.1
python run.py exp.name='prot_fcn_1_layers_0.1_contrastive_coef_lr_1e-3_l2_reg_0.1' method=protoformer dataset=swissprot lr=1e-3 +contrastive_coef=0.1 +protoformer_layer_num=1 weight_decay=0.1

python run.py exp.name='muris_fcn_1_layers_1_contrastive_coef_lr_1e-3_l2_reg_0.1' method=protoformer dataset=tabula_muris lr=1e-3 +contrastive_coef=1 +protoformer_layer_num=1 weight_decay=0.1
python run.py exp.name='muris_fcn_1_layers_0_contrastive_coef_lr_1e-3_l2_reg_0.1' method=protoformer dataset=tabula_muris lr=1e-3 +contrastive_coef=0 +protoformer_layer_num=1 weight_decay=0.1
python run.py exp.name='muris_fcn_1_layers_0.1_contrastive_coef_lr_1e-3_l2_reg_0.1' method=protoformer dataset=tabula_muris lr=1e-3 +contrastive_coef=0.1 +protoformer_layer_num=1 weight_decay=0.1