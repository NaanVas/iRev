# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import pandas as pd

from .pmf import ProbabilisticMatrixFatorization


class TARMF_ZeroShot(nn.Module):
    '''
    WWW 2018 TARMF_ZeroShot
    '''
    def __init__(self, opt):
        super(TARMF_ZeroShot, self).__init__()

        self.num_fea = 2
        self.opt = opt

        self.u_forward_gru = nn.GRU(input_size=768, hidden_size=1, bidirectional=True,batch_first=True)
        self.i_forward_gru = nn.GRU(input_size=768, hidden_size=1, bidirectional=True,batch_first=True)

        self.user_topic_att = nn.Sequential(
                            nn.Linear(2, 1),
                            nn.Tanh() 
        )
        self.item_topic_att = nn.Sequential(
                            nn.Linear(2, 1),
                            nn.Tanh()
        )

        self.user_proj = nn.Sequential(
                            nn.Linear(2, opt.id_emb_size),
                            nn.Tanh()
        )

        self.item_proj = nn.Sequential(
                            nn.Linear(2, opt.id_emb_size),
                            nn.Tanh()
        )

        self.user_features = torch.Tensor(np.load(f"checkpoints/user_features_pmf_{opt.dataset}_{opt.emb_opt}.npy")).to('cuda')
        self.item_features = torch.Tensor(np.load(f"checkpoints/item_features_pmf_{opt.dataset}_{opt.emb_opt}.npy")).to('cuda')

        self.reset_para()

    
    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas


        '''
            SEQUENCE ENCODING LAYER
            Eq 1 - 4
        '''
        u_emb, u_hn = self.u_forward_gru(user_doc)
        i_emb, i_hn = self.i_forward_gru(item_doc)

        '''
            TOPICAL ATTENTION LAYER
            Eq 5 - 7
        '''
        u_att = F.softmax(self.user_topic_att(u_emb), 1)
        i_att = F.softmax(self.item_topic_att(i_emb), 1)

        u_emb = u_emb * u_att
        i_emb = i_emb * i_att

        user_fea = torch.sum(u_emb, 1)
        item_fea = torch.sum(i_emb, 1)

        user_fea = self.user_proj(user_fea)
        item_fea = self.item_proj(item_fea)

        u_r_fea = self.user_features[uids]
        i_r_fea = self.item_features[iids]

        user_fea = torch.cat([user_fea, u_r_fea], dim=1)
        item_fea = torch.cat([item_fea, i_r_fea], dim=1)

        return user_fea, item_fea
    

    def reset_para(self):
        for x in [self.user_topic_att,
                  self.item_topic_att,
                  self.user_proj,
                  self.item_proj]:
            nn.init.uniform_(x[0].weight, -0.1, 0.1)
            nn.init.constant_(x[0].bias, 0.1)
