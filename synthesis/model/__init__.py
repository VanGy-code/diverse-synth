import itertools
import os
import pickle

import numpy as np
import torch


def count_hit(select, pairs2) -> int:
    pairs4 = list(itertools.combinations(select, 2))
    for pair in pairs4:
        if pair in pairs2:
            return 1
    return 0


def generate_pairs(num_layer, subset_length, next_layer):
    assert (num_layer > (subset_length + 2))

    result = []
    # combination of each 4 classes
    allpairs = list(itertools.combinations(range(num_layer), subset_length))
    # combination of each 2 classes
    allpairs2 = list(itertools.combinations(range(num_layer), 2))
    # randomly generate an ordered list
    order = list(np.random.permutation(len(allpairs)))

    for i in range(next_layer):
        if len(allpairs2) == 0:
            for j in order[0: next_layer - i]:
                result.append(list(allpairs[j]))
            break

        idx = np.random.randint(0, len(order) - 1)
        # select one subset, if subset_length = 4, select should be like (4, 5, 1, 6)
        select = allpairs[order[idx]]
        # if there is no relation between selected class, then redo select progress.
        while count_hit(select, allpairs2) < 1:
            idx = np.random.randint(0, len(order) - 1)
            select = allpairs[order[idx]]

        result.append(list(select))

        pairs4 = list(itertools.combinations(select, 2))
        for pair in pairs4:
            if pair in allpairs2:
                allpairs2.remove(pair)

        order.remove(order[idx])

    return torch.tensor(result)


def generate_pairs_reverse(num_class, num_each_class, pre_graph):
    result = []
    for i in range(num_class):
        class_result = []
        for j in range(len(pre_graph)):
            if i in pre_graph[j]:
                class_result.append(j)

        for _ in range(num_each_class):
            result.append(torch.tensor(class_result))
    return result


def dump_pairs(log_dir, num_class, num_each_class, model_config):
    dump_dir = f"{log_dir}/pairs"
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    if os.path.exists(f"{dump_dir}/pairs.pkl"):
        with open(f"{dump_dir}/pairs.pkl", 'rb') as f:
            pairs_dict = pickle.load(f)
        return pairs_dict

    pairs_dict = {}

    pairs_dict['pairs1'] = generate_pairs(
        num_class,
        model_config["sparse_num"],
        model_config["sparse_embedding1"]
    )

    pairs_dict['pairs2'] = generate_pairs(
        model_config["linear_embedding1"],
        model_config["sparse_num"],
        model_config["sparse_embedding2"]
    )

    pairs_dict['pairs3'] = generate_pairs(
        model_config["linear_embedding2"],
        model_config["sparse_num"],
        model_config["sparse_embedding3"]
    )

    pairs_dict['pairs_reverse1'] = generate_pairs_reverse(
        model_config["linear_embedding2"],
        1,
        pairs_dict['pairs3']
    )

    pairs_dict['pairs_reverse2'] = generate_pairs_reverse(
        model_config["linear_embedding1"],
        1,
        pairs_dict['pairs2']
    )

    pairs_dict['pairs_reverse3'] = generate_pairs_reverse(
        num_class,
        num_each_class,
        pairs_dict['pairs1']
    )

    with open(os.path.join(dump_dir, 'pairs.pkl'), 'wb') as f:
        pickle.dump(pairs_dict, f)

    return pairs_dict
