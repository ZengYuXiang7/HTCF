# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import *

import dgl as d
from dgl.nn.pytorch import SAGEConv
from utils.metamodel import MetaModel


class GraphMF(MetaModel):
    def __init__(self, user_num, serv_num, args):
        self.args = args
        super(GraphMF, self).__init__(user_num, serv_num, args)
        try:
            userg = pickle.load(open('./modules/models/baselines/userg.pk', 'rb'))
            servg = pickle.load(open('./modules/models/baselines/servg.pk', 'rb'))
        except:
            user_lookup, serv_lookup, userg, servg = create_graph()
            pickle.dump(userg, open('./modules/models/baselines/userg.pk', 'wb'))
            pickle.dump(servg, open('./modules/models/baselines/servg.pk', 'wb'))
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        self.order = args.order
        self.user_embeds = GraphSAGEConv(self.usergraph, args.dimension, args.order)
        self.item_embeds = GraphSAGEConv(self.servgraph, args.dimension, args.order)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, inputs, train):
        userIdx, itemIdx = inputs
        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.item_embeds(itemIdx)
        user_embeds = user_embeds.to(torch.float32)
        serv_embeds = serv_embeds.to(torch.float32)
        estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid()
        estimated = estimated.reshape(user_embeds.shape[0])
        return estimated.flatten()

    def prepare_test_model(self):
        pass

    def get_embeds_parameters(self):
        parameters = []
        for params in self.user_embeds.parameters():
            parameters += [params]
        for params in self.item_embeds.parameters():
            parameters += [params]
        return parameters

    def get_attention_parameters(self):
        parameters = []
        for params in self.user_attention.parameters():
            parameters += [params]
        for params in self.item_attention.parameters():
            parameters += [params]
        return parameters

    def get_mlp_parameters(self):
        parameters = []
        for params in self.layers.parameters():
            parameters += [params]
        return parameters


class GraphSAGEConv(torch.nn.Module):

    def __init__(self, graph, dim, order=3):
        super(GraphSAGEConv, self).__init__()
        self.order = order
        self.graph = graph
        self.embedding = torch.nn.Parameter(torch.Tensor(self.graph.number_of_nodes(), dim))
        torch.nn.init.kaiming_normal_(self.embedding)
        self.graph.ndata['L0'] = self.embedding
        self.layers = torch.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])

    def forward(self, uid):
        g = self.graph.to('cuda')
        feats = g.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            g.ndata[f'L{i + 1}'] = feats

        embeds = g.ndata[f'L{self.order}'][uid]
        return embeds

# GraphMF
def create_graph():
    userg = d.graph([])
    servg = d.graph([])
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()
    ufile = pd.read_csv('./datasets/原始数据/userlist_table.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    sfile = pd.read_csv('./datasets/原始数据/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()
    slines = slines

    for i in range(339):
        user_lookup.register('User', i)
    for j in range(5825):
        serv_lookup.register('Serv', j)

    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure)
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre)
    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr)
    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas)

    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))

    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[6])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        spr = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, spr):
            servg.add_edges(sid, spr)

    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)
    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:

    def __init__(self):
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category, value):
        # 添加进入类别
        self.__category.add(category)
        # 如果类别不存在若无则，则新增一个类别子树
        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        # 如果值不在全局索引中，则创建之，id += 1
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            # 如果值不存在与类别子树，则创建之
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        # 返回索引id
        return self.__inner_bag[value]

    def query_value(self, id):
        # 返回值
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)


