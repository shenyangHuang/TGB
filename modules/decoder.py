"""
Decoder modules for dynamic link prediction

"""

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class LinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        # h = F.log_softmax(h, dim=-1)
        return h


### for TKG:
class ConvTransE(torch.nn.Module):
    """
    https://github.com/Lee-zix/CEN/blob/main/src/decoder.py
    """
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, 
    feature_map_dropout=0, channels=50, kernel_size=3, sequence_len = 1, use_bias=True, model_name='REGCN'):

        super(ConvTransE, self).__init__()
        self.model_name = model_name #'REGCN' or 'CEN'
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.embedding_dim = embedding_dim

        # self.sequence_len = sequence_len

        self.conv_list = torch.nn.ModuleList()
        self.bn0_list = torch.nn.ModuleList()
        self.bn1_list = torch.nn.ModuleList()
        self.bn2_list = torch.nn.ModuleList()
        for _ in range(sequence_len):
            self.conv_list.append(torch.nn.Conv1d(2, channels, kernel_size, stride=1, 
            padding=int(math.floor(kernel_size / 2)))  ) # kernel size is odd, then padding = math.floor(kernel_size/2))
            self.bn0_list.append(torch.nn.BatchNorm1d(2))
            self.bn1_list.append( torch.nn.BatchNorm1d(channels))
            self.bn2_list.append(torch.nn.BatchNorm1d(embedding_dim)) 


        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, triplets, partial_embeding=None, samples_of_interest_emb=None):
        """ forward for ConvsTransE decoder that computes scores for given triples of question
        return: score_list: list of scores for each triple in the batch
        """
        score_list = []
        batch_size = len(triplets)
        if self.model_name == 'CEN': #CEN
            for idx in range(len(embedding)): # leng of test_graph
                if samples_of_interest_emb != None:
                    x= self.forward_inner(embedding[idx], emb_rel, triplets, idx, partial_embeding, samples_of_interest_emb[idx])     
                else:
                    x= self.forward_inner(embedding[idx], emb_rel, triplets, idx, partial_embeding, samples_of_interest_emb)
                score_list.append(x)
            return score_list
        else: #RE-GCN
            scores = self.forward_inner(embedding, emb_rel, triplets, 0, partial_embeding, samples_of_interest_emb)
            return scores 



    def forward_inner(self, embedding, emb_rel, triplets, idx=0, partial_embeding=None, samples_of_interest_emb=None):
        """ forward for ConvsTransE decoder that computes scores for given triples of question for each graph in the history of test graphs
        return: x: list of scores for each triple in the batch
        """
        batch_size = len(triplets)
        e1_embedded_all = F.tanh(embedding)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0_list[idx](stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv_list[idx](x)
        x = self.bn1_list[idx](x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2_list[idx](x)
        x = F.relu(x)
        if partial_embeding !=None:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        elif samples_of_interest_emb !=None: # added tgb team: predict only for nodes of interest
            x = torch.mm(x, F.tanh(samples_of_interest_emb).transpose(1, 0)) 
        else: #predict for all nodes
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))

        return x