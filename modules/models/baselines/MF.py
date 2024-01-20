# coding : utf-8
# Author : yuxiang Zeng

import torch
import numpy as np
import pandas as pd
import pickle as pk

from utils.metamodel import MetaModel


class Pure_Mf(MetaModel):
    def __init__(self, user_num, serv_num, args):
        super(Pure_Mf, self).__init__(user_num, serv_num, args)
        self.embed_user_GMF = torch.nn.Embedding(user_num, args.dimension)
        self.embed_item_GMF = torch.nn.Embedding(serv_num, args.dimension)
        self.predict_layer = torch.nn.Linear(args.dimension, 1)

    def forward(self, UserIdx, itemIdx):
        user_embed = self.embed_user_GMF(UserIdx)
        item_embed = self.embed_item_GMF(itemIdx)
        gmf_output = user_embed * item_embed
        prediction = self.predict_layer(gmf_output)
        return prediction.flatten()

