# PrototypeFormer
 EPFL CS502 DL Biomedicine 2023 Fall Project: Adapted Code Reproduction of PrototypeFormer for Transfer Learning in Biomedical Applications.

In the `fewshotbench` folder you can find the few shot learning benchmark with added PrototypeFormer method ([methods/protoformer.py](https://github.com/Alvorecer721/DL_Bio_PrototypeFormer/blob/main/fewshotbench/methods/protoformer.py)).

## Hyperparameter

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

## Usage

### Training

To reproduce our PrototypeFormer model with best parameters, we add two configuration files which fixes the parameters[protoformer_best_swissprot_params.yaml](https://github.com/Alvorecer721/DL_Bio_PrototypeFormer/blob/main/fewshotbench/conf/method/protoformer_best_swissprot_params.yaml) and [protoformer_best_tabula_muris_params.yaml](https://github.com/Alvorecer721/DL_Bio_PrototypeFormer/blob/main/fewshotbench/conf/method/protoformer_best_tabula_muris_params.yaml). You can reproduce our outcome with:

For SwissProt, using:

```
!python run.py \
    exp.name='reproduce best swissprot' \
    method=protonet \
    dataset=swissprot \
    ++n_shot=5
```

For Tabula Muris, using:

```
python run.py \
    exp.name='reproduce PrototypeFormer tabula muris' \
    method=protoformer_best_tabula_muris_params \
    dataset=tabula_muris \
    ++lr=0.002499525509356929 \
    ++weight_decay=0.00311156190841727 \
    ++n_shot=5
```

### Tuning

To tune this hyper-parameters as well as learning rate and weight decay, we used `optuna` library, please refer to [tuninig.py](https://github.com/Alvorecer721/DL_Bio_PrototypeFormer/blob/main/fewshotbench/tuning.py). You can tune with:

```
python tuning.py \
     --dataset "swissprot_no_backbone" \
     --embed 'protbert_emb'\
     --n_trials 30 \
     --stop_epoch 60 \
     --log_mode 'online'
```

We use ESM embedding for Swissprot as default, if you want to use Protbert as embedding, set `--embed 'protbert_emb'`. And if you don't want to keep your log on WandB, set `--log_mode 'offline'`.



### Protbert embedding

During the project, we also used the Protbert model to obtain different embeddings and compare the performance of the model while using different embeddings. Please, note, that due to the memory limitations, long (longer than 1000 characters) sequences were truncated. Overall, 7% of sequences from the SwissProt dataset were truncated while computing Protbert embeddings. You can find the code for the embedding creation in 
[protbert_embeddings.py](https://github.com/Alvorecer721/DL_Bio_PrototypeFormer/blob/main/protbert_embeddings.py). To get the Protbert embedding, use the following:

```
python protbert_embeddings.py
```

Then you will get the embedding at `data/swissprot/protber_emb`.

