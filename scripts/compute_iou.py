import argparse
import os

import numpy as np
import torch.utils.data
from scipy.io import savemat, loadmat


def compute_iou(rec1, rec2):
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    else:
        s_1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        s_2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        s_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return s_cross / (s_1 + s_2 - s_cross)


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type='bedroom', tag=None):
        self.path = f"{path}/{dataset_type}/{tag}/raw_data"
        self.dataset = sorted([
            f"{self.path}/{f}"
            for f in os.listdir(self.path)
            if f.endswith('.mat')
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset[idx]
        # return loadmat(path).get("boxes")
        return loadmat(path).get("boxes_aligned")


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
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
    parser.add_argument(
        "--path_to_renderings",
        default="/data/render_scene/scene-synth",
        help="Path to the folder containing the synthesized"
    )

    args = parser.parse_args(argv)

    dataset = BoxDataset(path=args.path_to_renderings, dataset_type=args.dataset_type, tag=args.tag)
    iou = []
    for i in range(len(dataset)):
        boxes = dataset[i]
        for j in range(0, len(boxes)):
            rec1 = boxes[j, [3, 2, 1, 0]]
            for k in range(j + 1, len(boxes)):
                rec2 = boxes[k, [3, 2, 1, 0]]
                iou.append(compute_iou(rec1, rec2))

    print(sum(iou) / len(iou))
    print(np.std(iou))


if __name__ == "__main__":
    main(None)
