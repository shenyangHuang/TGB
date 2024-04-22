import time
from tgb.linkproppred.thg_negative_generator import THGNegativeEdgeGenerator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


def main():
    r"""
    Generate negative edges for the validation or test phase
    """
    print("*** Negative Sample Generation ***")

    # setting the required parameters
    num_neg_e_per_pos = 20 #-1 
    neg_sample_strategy = "node-type-filtered"
    rnd_seed = 42


    name = "thgl-myket"
    dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()



    data_splits = {}
    data_splits['train'] = data[train_mask]
    data_splits['val'] = data[val_mask]
    data_splits['test'] = data[test_mask]

    min_node_idx = min(int(data.src.min()), int(data.dst.min()))
    max_node_idx = max(int(data.src.max()), int(data.dst.max()))

    neg_sampler = THGNegativeEdgeGenerator(
        dataset_name=name,
        first_node_id=min_node_idx,
        last_node_id=max_node_idx,
        node_type=dataset.node_type,
        num_neg_e=num_neg_e_per_pos,
        strategy=neg_sample_strategy,
        rnd_seed=rnd_seed,
        edge_data=data,
    )

   
    # generate evaluation set
    partial_path = "."
    # generate validation negative edge set
    start_time = time.time()
    split_mode = "val"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        pos_edges=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

    # generate test negative edge set
    start_time = time.time()
    split_mode = "test"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        pos_edges=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )


if __name__ == "__main__":
    main()
