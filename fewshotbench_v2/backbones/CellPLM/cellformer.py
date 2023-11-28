import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..embedder import OmicsEmbeddingLayer
from ..utils.mask import MaskBuilder, NullMaskBuilder, HiddenMaskBuilder
from ..encoder import setup_encoder
from ..decoder import setup_decoder
from ..latent import LatentModel, PreLatentNorm
from ..latent.adversarial import AdversarialLatentLayer
from ..objective import Objectives
from ..head import setup_head

class OmicsFormer(nn.Module):
    def __init__(self, gene_list, enc_hid, enc_layers, post_latent_dim, 
                  model_dropout=0.1,
                 activation='gelu', norm='layernorm', enc_head=8, 
                 pe_type='sin', cat_pe=True,
                 gene_emb=None, 
                input_covariate=False,batch_num=0,
                  **kwargs):
        super(OmicsFormer, self).__init__()

        self.embedder = OmicsEmbeddingLayer(gene_list, enc_hid, norm, activation, model_dropout,
                                            pe_type, cat_pe, gene_emb, inject_covariate=input_covariate, batch_num=batch_num)
        
       
        self.encoder = setup_encoder('transformer', enc_hid, enc_layers, model_dropout, activation, norm, enc_head)

        self.latent = LatentModel()
        
        
        self.latent.add_layer(type='vae', enc_hid=enc_hid, latent_dim=post_latent_dim)

    
        
        self.pre_latent_norm = PreLatentNorm('ln', enc_hid)
        # self.post_latent_norm = nn.LayerNorm(post_latent_dim, dataset_num)

    def forward(self, x_dict, input_gene_list=None, d_iter=False):
        if self.mask_type == 'input':
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict['h'] = self.embedder(x_dict, input_gene_list)
        if self.mask_type == 'hidden':
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict['h'] = self.encoder(x_dict)['hidden']
        x_dict['h'] = self.pre_latent_norm(x_dict)
        x_dict['h'], latent_loss = self.latent(x_dict)

        # x_dict['h'] = self.post_latent_norm(x_dict['h'])
        # if 'ecs' in x_dict:
        #     x_dict['h'] = self.latent_norm(x_dict['h'])

        if d_iter:
            return self.latent.d_train(x_dict)
        else:
            if self.head_type is not None:
                out_dict, loss = self.head(x_dict)
                out_dict['latent_loss'] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                out_dict['target_loss'] = loss.item()
            else:
                out_dict = self.decoder(x_dict)
                loss = latent_loss + self.objective(out_dict, x_dict) #/ 1e4
                out_dict['latent_loss'] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                out_dict['target_loss'] = loss.item() - out_dict['latent_loss']
            return out_dict, loss

    def nondisc_parameters(self):
        other_params = []
        for pname, p in self.named_parameters():
            if 'discriminator' not in pname:
                other_params += [p]
            else:
                print(pname)
        return other_params
