# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch
import numpy as np
import pandas as pd

from utils.metamodel import MetaModel

class CSMF(MetaModel):

    def __init__(self, user_num, serv_num, args):
        super(CSMF, self).__init__(user_num, serv_num, args)
        self.args = args
        self.UserList = pd.read_csv(args.path + 'userlist_idx.csv')
        self.ServList = pd.read_csv(args.path + 'wslist_idx.csv')
        self.UserEmbedding = torch.nn.Embedding(user_num, args.dimension)
        self.UserASEmbedding = torch.nn.Embedding(137, args.dimension)
        self.UserREEmbedding = torch.nn.Embedding(31, args.dimension)
        self.ServEmbedding = torch.nn.Embedding(serv_num, args.dimension)
        self.ServASEmbedding = torch.nn.Embedding(1603, args.dimension)
        self.ServREEmbedding = torch.nn.Embedding(74, args.dimension)
        self.ServPrEmbedding = torch.nn.Embedding(2699, args.dimension)

        self.user_norm = torch.nn.LayerNorm(args.dimension)
        self.serv_norm = torch.nn.LayerNorm(args.dimension)

        for layer in self.children():
            if isinstance(layer, torch.nn.Embedding):
                param_shape = layer.weight.shape
                layer.weight.data = torch.from_numpy(np.random.uniform(-1, 1, size=param_shape))

        if args.device == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.norm = torch.nn.LayerNorm(args.dimension)

    def forward(self, inputs, train):
        userIdx, servIdx = inputs
        user = np.array(userIdx.cpu(), dtype='int32')
        UserAS = torch.tensor(np.array(self.UserList['[AS]'][user]))
        UserRE = torch.tensor(np.array(self.UserList['[Country]'][user]))
        UserIdx = userIdx.to(self.device)
        UserAS = UserAS.to(self.device)
        UserRE = UserRE.to(self.device)
        user_embed = self.UserEmbedding(UserIdx)
        uas_embed = self.UserASEmbedding(UserAS)
        ure_embed = self.UserREEmbedding(UserRE)
        serv = np.array(servIdx.cpu(), dtype='int32')
        ServAS = torch.tensor(np.array(self.ServList['[AS]'][serv]))
        ServRE = torch.tensor(np.array(self.ServList['[Country]'][serv]))
        ServPr = torch.tensor(np.array(self.ServList['[Service Provider]'][serv]))
        user_vec = user_embed + uas_embed + ure_embed
        ServIdx = servIdx.to(self.device)
        ServAS = ServAS.to(self.device)
        ServRE = ServRE.to(self.device)
        ServPr = ServPr.to(self.device)
        serv_embed = self.ServEmbedding(ServIdx)
        sas_embed = self.ServASEmbedding(ServAS)
        sre_embed = self.ServREEmbedding(ServRE)
        spr_embed = self.ServPrEmbedding(ServPr)
        serv_vec = serv_embed + sas_embed + sre_embed + spr_embed
        user_vec = self.user_norm(user_vec.to(torch.float32))
        serv_vec = self.serv_norm(serv_vec.to(torch.float32))
        inner_prod = user_vec * serv_vec
        inner_prod = self.norm(inner_prod.float())
        tmp = torch.sum(inner_prod, -1)
        pred = tmp.sigmoid()

        return pred.flatten()

    def prepare_test_model(self):
        pass
