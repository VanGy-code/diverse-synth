import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data.dataset import T_co

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis.datasets import THREED_FRONT_FURNITURE, CSVSplitsBuilder
from synthesis.datasets.FUTURE import Furniture


class InfiniteDataset(IterableDataset):
    """
    Decorate any Dataset instance to provide an infinite IterableDataset
    version of it.
    """

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, dataset, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        N = len(self.dataset)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = N
        else:
            num_workers = worker_info.num_workers
            per_worker = (N + num_workers - 1) // num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, N)

        indices = np.arange(start, end)
        while True:
            if self.shuffle:
                np.random.shuffle(indices)
            for i in indices:
                yield self.dataset[i]


class ModelInfo(object):
    def __init__(self, model_info_data):

        self.model_info_data = model_info_data
        self._model_info = None
        # List to keep track of the different styles, themes
        self._styles = []
        self._themes = []
        self._categories = []
        self._super_categories = []
        self._materials = []

    @property
    def model_info(self):
        if self._model_info is None:
            self._model_info = {}
            # Create a dictionary of all models/assets in the dataset
            for m in self.model_info_data:
                # Keep track of the different styles
                if m["style"] not in self._styles and m["style"] is not None:
                    self._styles.append(m["style"])
                # Keep track of the different themes
                if m["theme"] not in self._themes and m["theme"] is not None:
                    self._themes.append(m["theme"])
                # Keep track of the different super-categories
                if m["super-category"] not in self._super_categories and m["super-category"] is not None:
                    self._super_categories.append(m["super-category"])
                # Keep track of the different categories
                if m["category"] not in self._categories and m["category"] is not None:
                    self._categories.append(m["category"])
                # Keep track of the different categories
                if m["material"] not in self._materials and m["material"] is not None:
                    self._materials.append(m["material"])

                super_cat = "unknown_super-category"
                cat = "unknown_category"

                if m["super-category"] is not None:
                    super_cat = m["super-category"].lower().replace(" / ", "/")

                if m["category"] is not None:
                    cat = m["category"].lower().replace(" / ", "/")
                    cat = THREED_FRONT_FURNITURE.get(cat)

                self._model_info[m["model_id"]] = Furniture(
                    super_cat,
                    cat,
                    m["style"],
                    m["theme"],
                    m["material"]
                )

        return self._model_info

    @property
    def styles(self):
        return self._styles

    @property
    def themes(self):
        return self._themes

    @property
    def materials(self):
        return self._materials

    @property
    def category(self):
        return self._categories

    @property
    def categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._categories])

    @property
    def super_categories(self):
        return set([
            s.lower().replace(" / ", "/")
            for s in self._super_categories
        ])

    @classmethod
    def load_file(cls, path_to_model_info):
        with open(path_to_model_info, "rb") as f:
            model_info = json.load(f)

        return cls(model_info)


class BaseDataset(Dataset):
    """
    Implements the interface for all datasets that consist of scenes.
    """

    def __init__(self, scenes):
        assert len(scenes) > 0
        self.scenes = scenes
        self._class_labels = {}

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

    @property
    def num_class(self):
        """
        Number of distinct objects in a Scene.
        """
        return len(self._class_labels)

    """
    Defined
    """

    @staticmethod
    def with_room(scene_type):
        """
        Filter type of room
        """

        def inner(scene):
            return scene if scene_type in scene.scene_type else False

        return inner

    @staticmethod
    def with_scene_ids(scene_ids):
        """
        Filter scene id to split dataset into train/test
        """

        def inner(scene):
            return scene if scene.scene_id in scene_ids else False

        return inner

    @staticmethod
    def with_object_types(objects):
        """
        Filter scene which reasonable contains furniture
        Example: Dining room should not have bed
        """

        def inner(scene):
            return (
                scene if all(b.label in objects for b in scene.bboxes)
                else False
            )

        return inner

    @staticmethod
    def contains_object_types(objects):
        """
        Filter scene which should contains special furniture
        Example: Bedroom should have at least one bed
        """

        def inner(scene):
            return (
                scene if any(b.label in objects for b in scene.bboxes)
                else False
            )

        return inner

    @staticmethod
    def with_generic_classes(box_types_map):
        """
        Unify the class of furniture
        """

        def inner(scene):
            for box in scene.bboxes:
                # Update the box label based on the box_types_map
                box.label = box_types_map[box.label]
            return scene

        return inner

    @staticmethod
    def without_object_types(objects):
        """
        Ignore rooms with special furniture
        """

        def inner(scene):
            return (
                False if any(b.label in objects for b in scene.bboxes)
                else scene
            )

        return inner

    @staticmethod
    def without_box_types(box_types):
        """
        Ignore rooms with special furniture
        """

        def inner(scene):
            for i in range(len(scene.bboxes) - 1, -1, -1):
                if scene.bboxes[i].label in box_types:
                    scene.bboxes.pop(i)
            return scene

        return inner

    @property
    def room_types(self):
        return set([si.scene_type for si in self.scenes])

    """
    Filter Dataset Error
    """

    @staticmethod
    def room_smaller_than_along_axis(max_size, axis=1):
        """
        the room size error
        """

        def inner(scene):
            return scene if scene.bbox[1][axis] <= max_size else False

        return inner

    @staticmethod
    def room_larger_than_along_axis(min_size, axis=1):
        """
            the room size error
        """

        def inner(scene):
            return scene if scene.bbox[0][axis] >= min_size else False

        return inner

    @staticmethod
    def at_least_boxes(n):
        """
        Ignore scene with few of furniture
        """

        def inner(scene):
            return scene if len(scene.bboxes) >= n else False

        return inner

    @staticmethod
    def at_most_boxes(n):
        """
            Ignore scene with too much of furniture
        """

        def inner(scene):
            return scene if len(scene.bboxes) <= n else False

        return inner

    @staticmethod
    def with_valid_scene_ids(invalid_scene_ids):
        """
        Filter invalid scene
        """

        def inner(scene):
            return scene if scene.scene_id not in invalid_scene_ids else False

        return inner

    @staticmethod
    def with_valid_bbox_jids(invalid_bbox_jds):
        """
        Filter furniture with invalid bbox
        """

        def inner(scene):
            return (
                False if any(b.model_jid in invalid_bbox_jds for b in scene.bboxes)
                else scene
            )

        return inner

    @staticmethod
    def with_valid_boxes(box_types):
        def inner(scene):
            for i in range(len(scene.bboxes) - 1, -1, -1):
                if scene.bboxes[i].label not in box_types:
                    scene.bboxes.pop(i)
            return scene

        return inner

    @staticmethod
    def floor_plan_with_limits(limit_x, limit_y, axis=None):
        if axis is None:
            axis = [0, 2]

        def inner(scene):
            min_bbox, max_bbox = scene.floor_plan_bbox
            t_x = max_bbox[axis[0]] - min_bbox[axis[0]]
            t_y = max_bbox[axis[1]] - min_bbox[axis[1]]
            if t_x <= limit_x and t_y <= limit_y:
                return scene

        return inner

    @staticmethod
    def filter_compose(*filters):
        def inner(scene):
            s = scene
            fs = iter(filters)
            try:
                while s:
                    s = next(fs)(s)
            except StopIteration:
                pass
            return s

        return inner

    @property
    def count_objects_in_rooms(self):
        return Counter([len(si.bboxes) for si in self.scenes])

    def post_process(self, s):
        return s

def filter_function(config, split=None, without_lamps=False):
    if split is None:
        split = ["train", "val"]
    print("Applying {} filtering".format(config["room_type_filter"]))
    if config["room_type_filter"] == "no_filtering":
        return lambda s: s

    # Parse the list of the invalid scene ids
    with open(config["path_to_invalid_scene_ids"], "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    # Parse the list of the invalid bounding boxes
    with open(config["path_to_invalid_bbox_jids"], "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    # Make the train/test/validation splits
    splits_builder = CSVSplitsBuilder(config["annotation_file"])
    split_scene_ids = splits_builder.get_splits(split)

    if "bedroom" in config["room_type_filter"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("bed"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(13),
            BaseDataset.with_object_types(
                list(THREED_FRONT_FURNITURE.values())
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.contains_object_types(
                ["bed"]
            ),
            BaseDataset.room_smaller_than_along_axis(6.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(8, 8, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "livingroom" in config["room_type_filter"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("living"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(21),
            BaseDataset.with_object_types(
                list(THREED_FRONT_FURNITURE.values())
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["lamp", "sofa"]
                if without_lamps else [""]
            ),
            BaseDataset.without_box_types(
                ["bed"]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "diningroom" in config["room_type_filter"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("dining"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(21),
            BaseDataset.with_object_types(
                list(THREED_FRONT_FURNITURE.values())
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12, axis=[0, 2]),
            BaseDataset.contains_object_types(["dining_chair", "dinging_table"]),
            BaseDataset.without_box_types(
                ["lamp", "dining_table"]
                if without_lamps else [""]
            ),
            BaseDataset.without_box_types(
                ["bed"]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "library" in config["room_type_filter"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("library"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.with_object_types(
                list(THREED_FRONT_FURNITURE.values())
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(6, 6, axis=[0, 2]),
            BaseDataset.contains_object_types(
                ["shelf"]
            ),
            BaseDataset.without_box_types(
                ["lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.without_box_types(
                ["bed"]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    else:
        return lambda s: s if len(s.bboxes) > 0 else False