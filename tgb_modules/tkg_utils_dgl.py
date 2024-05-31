
import dgl
import torch
import numpy as np


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
