import argparse
import logging
import os
import sys

import numpy as np
import torch

from scipy.io import savemat, loadmat

def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the KL-divergence between the object category "
                     "distributions of real and synthesized scenes")
    )

    parser.add_argument(
        "--output_directory",
        default="/data/kl_object_category/",
        help="Path to the output directory"
    )

    parser.add_argument(
        "--path_to_renderings",
        default="/data/render_scene/scene-synth",
        help="Path to the folder containing the synthesized"
    )

    parser.add_argument(
        "--dataset_type",
        default="bedroom",
        choices=[
            "bedroom",
            "livingroom",
            "diningroom",
            "library"
        ],
        help="The type of dataset filtering to be used"
    )

    parser.add_argument(
        "--tag",
        default="baseline",
    )

    args = parser.parse_args(argv)

    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    ground_truth_scenes = []
    synthesized_scenes = []
    processed_scenes = []

    data_path = f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/raw_data"
    for f in os.listdir(data_path):
        if f.endswith('.mat'):
            data = loadmat(f"{data_path}/{f}")
            # print(data)
            synthesized_scenes.append(np.array(data['boxes'][..., -1], dtype=int))
            processed_scenes.append(np.array(data['boxes_aligned'][..., -1], dtype=int))

    raw_data_path = f"{args.path_to_renderings}/raw_{args.dataset_type}/raw_data"
    for f in os.listdir(raw_data_path):
        if f.endswith('.mat'):
            data = loadmat(f"{raw_data_path}/{f}")
            ground_truth_scenes.append(np.array(data['boxes'][..., -1], dtype=int))

    np.random.shuffle(ground_truth_scenes)
    gt_class_labels = np.zeros(23)

    for d in ground_truth_scenes[0:1000]:
        for index in d:
            gt_class_labels[index] += 1

    gt_class_labels = gt_class_labels / sum([d.sum(0) for d in gt_class_labels])

    syn_class_labels = np.zeros(23)
    for d in synthesized_scenes:
        for index in d:
            syn_class_labels[index] += 1
    syn_class_labels = syn_class_labels / sum([d.sum(0) for d in syn_class_labels])

    processed_class_labels = np.zeros(23)
    for d in processed_scenes:
        for index in d:
            processed_class_labels[index] += 1
    processed_class_labels = processed_class_labels / sum([d.sum(0) for d in processed_class_labels])

    assert 0.9999 <= gt_class_labels.sum() <= 1.0001
    assert 0.9999 <= syn_class_labels.sum() <= 1.0001
    assert 0.9999 <= processed_class_labels.sum() <= 1.0001

    stats = {}
    stats["gt_vs_syn_class_labels"] = categorical_kl(gt_class_labels, syn_class_labels)
    stats["syn_vs_pro_class_labels"] = categorical_kl(syn_class_labels, processed_class_labels)
    stats["gt_vs_pro_class_labels"] = categorical_kl(gt_class_labels, processed_class_labels)
    print(stats)


if __name__ == "__main__":
    main(sys.argv[1:])
