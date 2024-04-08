import numpy as np
import torch
import dgl

def split_by_time(data):
    """
    https://github.com/Lee-zix/CEN/blob/main/rgcn/utils.py
    create list where each entry has an entry with all triples for this timestep
    """
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]

        if latest_t != t:  
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])

    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    return snapshot_list

def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
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
    # src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    # rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    if use_cuda:
        g = g.to(gpu)
    return g
