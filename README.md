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
|dimension of ffn in encoder |                ffn_dim|


To tune this hyper-parameters as well as learning rate and weight decay, we used `optuna` library, please refer to `tuninig.py`.

During the project we also used protbert model to obtain different embeddings and compare the performance of the model while using different embeddings. Please, note, that due to the memory limitations, long (longer than 1000 characters) sequences were truncaetd. Overall, 7% of sequences from swissprot dataset were truncated while computing protbert embeddings.
