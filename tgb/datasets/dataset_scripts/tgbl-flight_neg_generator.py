import time


from tgb.linkproppred.negative_generator import NegativeEdgeGenerator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset



def main():
    r"""
    Generate negative edges for the validation or test phase
    """
    print("*** Negative Sample Generation ***")

    # setting the required parameters
    num_neg_e_per_pos = 20
    neg_sample_strategy = "hist_rnd" #"rnd"
    rnd_seed = 42


    name = "tgbl-flight"
    dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()

    data_splits = {}
    data_splits['train'] = data[train_mask]
    data_splits['val'] = data[val_mask]
    data_splits['test'] = data[test_mask]

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # After successfully loading the dataset...
    if neg_sample_strategy == "hist_rnd":
        historical_data = data_splits["train"]
    else:
        historical_data = None

    neg_sampler = NegativeEdgeGenerator(
        dataset_name=name,
        first_dst_id=min_dst_idx,
        last_dst_id=max_dst_idx,
        num_neg_e=num_neg_e_per_pos,
        strategy=neg_sample_strategy,
        rnd_seed=rnd_seed,
        historical_data=historical_data,
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
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
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
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )


if __name__ == "__main__":
    main()
