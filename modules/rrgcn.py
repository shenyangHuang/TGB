"""
https://github.com/Lee-zix/CEN/blob/main/src/rrgcn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.rgcn_layers import UnionRGCNLayer, RGCNBlockLayer
from modules.rgcn_model import BaseRGCN
from modules.decoder import ConvTransE
import numpy as np
class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, 
                             rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            for i, layer in enumerate(self.layers):
                layer(g, [])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class RecurrentRGCNCEN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0):
        super(RecurrentRGCNCEN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.gpu = gpu

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float() #TODO: correct number?
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.loss_e = torch.nn.CrossEntropyLoss()


        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)    
        
      
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout, 
                                         sequence_len=self.sequence_len, model_name='CEN')
        else:
            raise NotImplementedError 


    def forward(self, g_list, use_cuda):
        evolve_embs = []
        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            current_h = self.rgcn.forward(g, self.h)
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            self.h = F.normalize(self.h)
            evolve_embs.append(self.h)
        return evolve_embs, self.emb_rel

    def predict(self, test_graph, test_triplets, use_cuda, neg_samples_batch=None, pos_samples_batch=None, 
                evaluator=None, metric=None):
        with torch.no_grad():
            scores = torch.zeros(len(test_triplets), self.num_ents).cuda()
            evolve_embeddings = []
            for idx in range(len(test_graph)):
                evolve_embs, r_emb = self.forward(test_graph[idx:], use_cuda)
                evolve_embeddings.append(evolve_embs[-1])
            evolve_embeddings.reverse()

            if neg_samples_batch != None: # added by tgb team
                perf_list = []
                hits_list = []
                for query_id, query in enumerate(neg_samples_batch): # for each sample separately
                    pos = pos_samples_batch[query_id]
                    neg = torch.tensor(query).to(pos.device)
                    all =torch.cat((pos.unsqueeze(0), neg), dim=0)
                    score_list = self.decoder_ob.forward(evolve_embeddings, r_emb, test_triplets[query_id].unsqueeze(0),
                            samples_of_interest_emb= [evolve_embeddings[i][all] for i in range(len(evolve_embeddings))])
                    score_list = [_.unsqueeze(2) for _ in score_list]
                    scores_b = torch.cat(score_list, dim=2)
                    scores_b = torch.softmax(scores_b, dim=1)
                    scores_b = torch.sum(scores_b, dim=-1)
                    # compute MRR
                    input_dict = {
                        "y_pred_pos": np.array([scores_b[0,0].cpu()]),
                        "y_pred_neg": np.array(scores_b[0,1:].cpu()),
                        "eval_metric": [metric],
                    }
                    prediction_perf = evaluator.eval(input_dict)
                    perf_list.append(prediction_perf[metric])
                    hits_list.append(prediction_perf['hits@10'])

            else:
                score_list = self.decoder_ob.forward(evolve_embeddings, r_emb, test_triplets, mode="test")

                score_list = [_.unsqueeze(2) for _ in score_list]
                scores = torch.cat(score_list, dim=2)
                scores = torch.softmax(scores, dim=1)
                scores = torch.sum(scores, dim=-1)

            return scores, perf_list, hits_list

    def get_ft_loss(self, glist, triple_list,  use_cuda):
        #"""
        #:param glist:
        #:param triplets:
        #:param use_cuda:
        #:return:
        #"""
        glist = [g.to(self.gpu) for g in glist]
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # for step, triples in enumerate(triple_list):
        evolve_embeddings = []
        for idx in range(len(glist)):
            evolve_embs, r_emb = self.forward(glist[idx:], use_cuda)
            evolve_embeddings.append(evolve_embs[-1])
        evolve_embeddings.reverse()
        scores_ob = self.decoder_ob.forward(evolve_embeddings, r_emb, triple_list[-1])#.view(-1, self.num_ents)
        for idx in range(len(glist)):
            loss_ent += self.loss_e(scores_ob[idx], triple_list[-1][:, 2])
        return loss_ent

    def get_loss(self, glist, triples, prev_model, use_cuda):
        """
        :param glist:
        :param triplets:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        evolve_embeddings = []
        for idx in range(len(glist)):
            evolve_embs, r_emb = self.forward(glist[idx:], use_cuda)
            evolve_embeddings.append(evolve_embs[-1])
        evolve_embeddings.reverse()
        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(evolve_embeddings, r_emb, triples)#.view(-1, self.num_ents)
            for idx in range(len(glist)):
                loss_ent += self.loss_e(scores_ob[idx], triples[:, 2])
        return loss_ent
    


class RecurrentRGCNREGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCNREGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)


        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                                 

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []
        # a = True
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
            a = torch.isnan(F.normalize(static_emb)).any() or torch.isinf(static_emb).any()
            if a ==True:
                print("static_emb is nan")
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h) #, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda, neg_samples_batch=None,
                pos_samples_batch=None, evaluator=None, metric=None):
        perf_list = [None]*len(neg_samples_batch)
        hits_list = [None]*len(neg_samples_batch)
        with torch.no_grad():
            # inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            # inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples =test_triplets # torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            if neg_samples_batch != None: # added by tgb team
                perf_list = []
                hits_list = []
                for query_id, query in enumerate(neg_samples_batch): # for each sample separately
                    pos = pos_samples_batch[query_id]
                    neg = torch.tensor(query).to(pos.device)
                    all =torch.cat((pos.unsqueeze(0), neg), dim=0)
                    score = self.decoder_ob.forward(embedding, r_emb, test_triplets[query_id].unsqueeze(0),
                            samples_of_interest_emb=embedding[all] )
                    # compute MRR
                    input_dict = {
                        "y_pred_pos": np.array([score[0,0].cpu()]),
                        "y_pred_neg": np.array(score[0,1:].cpu()),
                        "eval_metric": [metric],
                    }
                    prediction_perf = evaluator.eval(input_dict)
                    perf_list.append(prediction_perf[metric])
                    hits_list.append(prediction_perf['hits@10'])
            else:
                score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            # score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return score, perf_list, hits_list
        
    def get_mask_nonzero(self, static_embedding):
        """ Each element of this resulting tensor will be True if the sum of the corresponding row in 
        static_emb is not zero, and False otherwise
        """
        mask = torch.sum(static_embedding, dim=1) != 0
        return mask

    def get_loss(self, glist, triples, static_graph, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # inverse_triples = triples[:, [2, 1, 0]]
        # inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = triples #torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        a= torch.isnan(F.normalize(evolve_emb)).any() or torch.isinf(evolve_emb).any()
                        if a ==True:
                            print("evolve_emb is nan")
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                        a = torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any()
                        if a ==True:
                            print("sim_matrix is nan")
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        non_zero_mask = c != 0

                        # Initialize b_sim_matrix with zeros (or another appropriate value)
                        sim_matrix = torch.zeros_like(sim_matrix)

                        # Perform division only where c is not zero
                        sim_matrix[non_zero_mask] = sim_matrix[non_zero_mask] / c[non_zero_mask]
                        # sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    # mask = self.get_mask_nonzero(static_emb) #modified! to only consider non-zero rows
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_static