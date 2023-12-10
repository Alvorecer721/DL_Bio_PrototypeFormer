# PrototypeFormer
 EPFL CS502 DL Biomedicine 2023 Fall Project: Adapted Code Reproduction of PrototypeFormer for Transfer Learning in Biomedical Applications.

In the fewshotbench folder you can find the few shot learning benchmark with added PrototypeFormer method (`methods/protoformer.py`).

There are a lot of hyperparameters that you can tune with PrototypeFormer, such as   
|Hyperparameter              |                   name|
|----------------------------|-----------------------|
|number of transformer layers|  protoformer_layer_num|
|number of transformer heads |   protoformer_head_num|  
|contrastive loss type       |       contrastive_loss|
|contrastive loss coeficient |       contrastive_coef|
|number of sub-supports      |            sub_support|
|dropout                     |                dropout|
|layer norm before layers    |     encoder_norm_first|


To see how to run an experiment refer to `run.sh`.