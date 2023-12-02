#!/bin/sh

cd fewshotbench

python run.py exp.name='prot_no_backbone_1_layer' method=protoformer dataset=swissprot lr=1e-5 +protoformer_is_feature=1 +protoformer_layer_num=1 
python run.py exp.name='prot_no_backbone_3_layers' method=protoformer dataset=swissprot lr=1e-5 +protoformer_is_feature=1 +protoformer_layer_num=3
python run.py exp.name='prot_fcn_backbone_1_layer' method=protoformer dataset=swissprot lr=1e-4 +protoformer_is_feature=0 +protoformer_layer_num=1 
python run.py exp.name='prot_fcn_backbone_3_layers' method=protoformer dataset=swissprot lr=1e-4 +protoformer_is_feature=0 +protoformer_layer_num=3