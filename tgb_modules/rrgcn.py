"""
https://github.com/Lee-zix/CEN/blob/main/src/rrgcn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgb_modules.rgcn_layers import UnionRGCNLayer, RGCNBlockLayer
from tgb_modules.rgcn_model import BaseRGCN
from tgb_modules.decoder import ConvTransE
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
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
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
            raise NotImplementedError

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0):
        super(RecurrentRGCN, self).__init__()

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

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
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
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
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
                # model_idx = len(test_graph) - idx - 1 
                evolve_embs, r_emb = self.forward(test_graph[idx:], use_cuda)
                evolve_embeddings.append(evolve_embs[-1])
            evolve_embeddings.reverse()

            # if neg_samples_batch != None:
            #     partial_embedding = []
            if neg_samples_batch != None:
            #     perf_list = []
            #     import numpy as np
            #     metric= 'mrr'
                perf_list = []
                for query_id, query in enumerate(neg_samples_batch):
                    pos = pos_samples_batch[query_id]
                    neg = torch.tensor(query).to(pos.device)
                    all =torch.cat((pos.unsqueeze(0), neg), dim=0)
                    score_list = self.decoder_ob.forward(evolve_embeddings, r_emb, test_triplets[query_id].unsqueeze(0),
                                mode="test", 
                                neg_samples_embd= [evolve_embeddings[i][all] for i in range(len(evolve_embeddings))])
                    score_list = [_.unsqueeze(2) for _ in score_list]
                    scores_b = torch.cat(score_list, dim=2)
                    scores_b = torch.softmax(scores_b, dim=1)
                                # compute MRR
                    input_dict = {
                        "y_pred_pos": np.array([scores_b[0, :].squeeze(dim=-1).cpu()]),
                        "y_pred_neg": np.array(scores_b[1:, :].squeeze(dim=-1).cpu()),
                        "eval_metric": [metric],
                    }
                    perf_list.append(evaluator.eval(input_dict)[metric])

            else:
                score_list = self.decoder_ob.forward(evolve_embeddings, r_emb, test_triplets, mode="test")

                score_list = [_.unsqueeze(2) for _ in score_list]
                scores = torch.cat(score_list, dim=2)
                scores = torch.softmax(scores, dim=1)
                # scores = torch.max(scores, dim=2)[0]

                scores = torch.sum(scores, dim=-1)
                score_rel = torch.zeros_like(scores)
            return scores, perf_list

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