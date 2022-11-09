import os
import sys

import numpy as np
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis.datasets.Utils import parse_threed_future_models


class FutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects".format(
            len(self),
        )

    def __getitem__(self, idx):
        return self.objects[idx]
    
    @property
    def labels(self):
        return list(set([oi.label for oi in self.objects]))

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]

    def get_closest_furniture_to_box(self, query_label, query_size, invalid_list):
        objects = self._filter_objects_by_label(query_label)
        mses = {}
        for i, oi in enumerate(objects):
            if oi.model_jid in invalid_list:
                continue
            scale = oi.size * oi.scale * 2
            size = np.array([scale[2], scale[0], scale[1]])
            mses[oi] = np.sum((size - query_size) ** 2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (
                    (oi.size[0] - query_size[0]) ** 2 +
                    (oi.size[2] - query_size[1]) ** 2
            )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    @classmethod
    def from_dataset_directory(
            cls, dataset_directory, path_to_model_info, path_to_models
    ):
        objects = parse_threed_future_models(
            dataset_directory, path_to_models, path_to_model_info
        )
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
