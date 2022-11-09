"""
3D-FRONT Dataset
"""
import json
import os
import pickle
from functools import lru_cache
import itertools

import numpy as np
from PIL import Image

from tqdm import tqdm
import time

from synthesis.datasets import THREED_FRONT_FURNITURE
from synthesis.datasets.Common import BaseDataset
from synthesis.datasets.Room import Room, CachedRoom
from synthesis.datasets.Utils import parse_threed_front_scenes


class Front(BaseDataset):
    """
    Container for the scenes in the 3D-FRONT dataset.
    """

    def __init__(self, scenes, furniture_limit=4, bounds=None):
        super().__init__(scenes)
        # raise except when list of Room objects do not contain Room
        assert isinstance(self.scenes[0], Room)
        self._object_types = None
        self._room_types = None

        self._class_order = dict()
        self._furniture_limit = furniture_limit
        self._bbox = None

        self._sizes = self._centroids = self._angles = None

        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]
    
    def __str__(self):
        return "Dataset contains {} scenes".format(
            len(self.scenes)
        )

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
            len(self.scenes), self.num_class
        )

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000] * 3)
        _size_max = np.array([-10000000] * 3)
        _centroid_min = np.array([10000000] * 3)
        _centroid_max = np.array([-10000000] * 3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])
        for s in self.scenes:
            for f in s.bboxes:
                if np.any(f.size > 5):
                    print(s.scene_id, f.size, f.model_uid, f.scale)
                centroid = self._centroid(f, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)
        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles

    @property
    def bbox(self):
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    @property
    def bounds(self):
        return {"translations": self.centroids, "sizes": self.sizes, "angles": self.angles}

    @property
    def class_order(self):
        """
            Number furniture
            Example: {'wardrobe': 0, 'double_bed': 1, 'table': 2, 'cabinet': 3, 'nightstand': 4, 'ceiling_lamp': 5}
        """

        index = 0
        for d in [THREED_FRONT_FURNITURE]:
            for k, v in d.items():
                if v not in self._class_order.keys():
                    self._class_order[v] = index
                    index += 1
        return self._class_order

    @property
    def class_labels(self):
        return list(self.class_order.keys())

    @property
    def num_class(self):
        return len(self.class_labels)

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def furniture_limit(self):
        return self._furniture_limit

    @classmethod
    def load_dataset(cls, dataset_directory, path_to_model_info,
                     path_to_models, path_to_room_masks_dir=None,
                     path_to_bounds=None, room_type_filter=lambda s: s, ):
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)
        
        scenes = parse_threed_front_scenes(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir
        )
        results = [s for i, s in enumerate(tqdm(scenes)) if room_type_filter(s)]
        return cls(results, bounds=None)

class CachedFront(Front):
    def __init__(self, base_dir, config, scene_ids):
        self._base_dir = base_dir
        self.config = config

        self._parse_train_stats(config["train_stats"])

        self._tags = sorted([
            oi
            for oi in os.listdir(self._base_dir)
            if oi.split("_")[1] in scene_ids
        ])

        self._path_to_rooms = sorted([
            f"{self._base_dir}/{pi}"
            for pi in self._tags
        ])

        rendered_scene = "rendered_scene.png"

        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )

        if os.path.isfile(path_to_rendered_scene):
            self._path_to_renders = sorted([
                os.path.join(self._base_dir, pi, rendered_scene)
                for pi in self._tags
            ])

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    @lru_cache(maxsize=32)
    def __getitem__(self, i):
        D = np.load(self._path_to_rooms[i])
        return CachedRoom(
            scene_id=D["scene_id"],
            floor_plan_vertices=D["floor_plan_vertices"],
            floor_plan_faces=D["floor_plan_faces"],
            floor_plan_centroid=D["floor_plan_centroid"],
            class_labels=D["class_id"],
            translations=D["translations"],
            sizes=D["sizes"],
            angles=D["angles"],
            image_path=self._path_to_renders[i],
        )

    def get_room_params(self, i):
        D = np.load(os.path.join(self._path_to_rooms[i], "boxes.npz"))
        try:
            with open(f"{self._path_to_rooms[i]}/rel.pkl", 'rb') as f:
                x_rel = pickle.load(f)
            return {
                "scene_id": D["scene_id"],
                "x_abs": D["x_abs"],
                "x_rel": x_rel,
            }
        except EOFError:
            print(f"{self._path_to_rooms[i]}/rel.pkl is not find")
            raise EOFError

    def __len__(self):
        return len(self._path_to_rooms)

    def __str__(self):
        return "Dataset contains {} scenes with discrete types".format(
            len(self)
        )

    def _parse_train_stats(self, train_stats):
        with open(f"{self._base_dir}/{train_stats}", "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3]),
            np.array(self._centroids[3:])
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._class_order = train_stats["class_order"]
        self._furniture_limit = train_stats["furniture_limit"]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def num_class(self):
        return len(self.class_labels)

    @property
    def class_order(self):
        return self._class_order

    @property
    def furniture_limit(self):
        return self._furniture_limit
