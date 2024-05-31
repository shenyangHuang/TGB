
import dgl
import torch
import numpy as np
from collections import defaultdict

def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu, mode='dyn'):
    """
    https://github.com/Lee-zix/CEN/blob/main/rgcn/utils.py
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param use_cuda:
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    if mode =='static':
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + num_rels))
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    #g.ndata['original_id'] = np.unique(np.concatenate((np.unique(triples[:,0]), np.unique(triples[:,2]))))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    #node_id =torch.arange(0, g.num_nodes(), dtype=torch.long).view(-1, 1) #updated to deal with the fact that ot only the first k nodes of our graph have static infos
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)


    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len

    if use_cuda:
        g = g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g


def r2e(triplets, num_rels):
    """ get the mapping from relation to entities helper function for build_sub_graph()
    returns: 
    uniq_r: set of unique relations
    r_len: list of tuples, where each tuple is the start and end index of entities for a relation
    e_idx: indices of entities"""
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    # uniq_r = np.concatenate((uniq_r, uniq_r+num_rels)) #we already have the inverse triples
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx