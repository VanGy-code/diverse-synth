import copy
from functools import lru_cache

import numpy as np
from torch.utils.data import Dataset

from synthesis.datasets import CSVSplitsBuilder
from synthesis.datasets.FRONT import CachedFront, Front


def get_encoded_dataset(
        config,
        filter_fn=lambda s: s,
        path_to_bounds=None,
        split=["train", "val"]
):
    _, encoding = get_dataset_raw_and_encoded(
        config, filter_fn, path_to_bounds, split
    )
    return encoding


def get_raw_dataset(
        config,
        filter_fn=lambda s: s,
        path_to_bounds=None,
        split=["train", "val"]
):
    dataset_type = config["dataset_type"]
    if "cached" in dataset_type:
        # Make the train/test/validation splits
        splits_builder = CSVSplitsBuilder(config["annotation_file"])
        split_scene_ids = splits_builder.get_splits(split)
        dataset = CachedFront(
            config["dataset_directory"],
            config=config,
            scene_ids=split_scene_ids
        )
    else:
        dataset = Front.load_dataset(
            config["dataset_directory"],
            config["path_to_model_info"],
            config["path_to_models"],
            config["path_to_room_masks_dir"],
            path_to_bounds,
            filter_fn
        )
    return dataset


def get_dataset_raw_and_encoded(
        config,
        filter_fn=lambda s: s,
        path_to_bounds=None,
        split=["train", "val"]
):
    dataset = get_raw_dataset(config, filter_fn, path_to_bounds, split=split)
    encoding = FrontFactory.encode_dataset(
        config.get("encoding_type"),
        dataset
    )

    return dataset, encoding


class DatasetDecoratorBase(Dataset):
    """
    A base class that helps us implement decorators for ThreeDFront-like datasets.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def num_class(self):
        return self._dataset.num_class

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_order(self):
        return self._dataset.class_order

    @property
    def furniture_limit(self):
        return self._dataset.furniture_limit

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxDataset(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    @lru_cache(maxsize=16)
    def get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        return scene.bboxes


class DataEncoder(BoxDataset):
    """
    DataEncoder is a wrapper for all datasets we have
    """

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    @property
    def property_type(self):
        raise NotImplementedError()

class ClassIndexEncoder(DataEncoder):
    """
    Implement the encoding for the class labels.
    Example:
    [ class_id ]
    """

    @property
    def property_type(self):
        return "class_id"

    def __getitem__(self, idx):
        # Get the scene
        furniture_in_room = self.get_boxes(idx)

        # sequence length
        L = len(furniture_in_room)
        classes = np.zeros(L, dtype=np.int8)

        # build class dict
        for i, furniture in enumerate(furniture_in_room):
            classes[i] = self.class_order.get(furniture.label, -1)
        return classes

    @property
    def bbox_dims(self):
        return 1


class TranslationEncoder(DataEncoder):
    """
    Translation Encoder : [x, y, z]
    """

    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        furniture_in_room = self.get_boxes(idx)
        L = len(furniture_in_room)  # sequence length

        translations = np.zeros((L, 3), dtype=np.float32)
        for i, furniture in enumerate(furniture_in_room):
            translations[i] = furniture.centroid()
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(DataEncoder):
    """
    Example: [scale_x. scale_y, scale_z]
    """

    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        furniture_in_room = self.get_boxes(idx)
        L = len(furniture_in_room)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, furniture in enumerate(furniture_in_room):
            sizes[i] = furniture.half_size * furniture.scale * 2
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncoder(DataEncoder):
    """
    Example: [direction_x, direction_y, direction_z]
    """

    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        furniture_in_room = self.get_boxes(idx)
        L = len(furniture_in_room)
        # Get the rotation matrix for the current scene
        rotation = np.zeros((L, 3), dtype=np.float32)
        for i, furniture in enumerate(furniture_in_room):
            rotation[i] = furniture.direction()
        return rotation

    @property
    def bbox_dims(self):
        return 3


class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return 10

    def __getitem__(self, idx):
        sample_dict = {}
        layout = np.zeros((self.num_class * self.furniture_limit, 10))
        layout_dict = {i: [] for i in range(self.num_class)}

        for dataset in self._datasets:
            sample_dict[dataset.property_type] = dataset[idx]

        for index in range(len(sample_dict["class_id"])):
            if sample_dict["class_id"][index] != -1:
                # Furniture exist if != -1
                layout_dict[sample_dict["class_id"][index]].append(
                    np.concatenate(
                        [
                            [sample_dict["angles"][index][2], sample_dict["angles"][index][0],
                             sample_dict["angles"][index][1]],
                            [sample_dict["translations"][index][2], sample_dict["translations"][index][0],
                             sample_dict["translations"][index][1]],
                            [sample_dict["sizes"][index][0], sample_dict["sizes"][index][2],
                             sample_dict["sizes"][index][1]],
                            [1]
                        ]
                    )
                )

        for k, v in layout_dict.items():
            if len(v) != 0:
                lv = np.minimum(len(v), self.furniture_limit)
                layout[k * self.furniture_limit: k * self.furniture_limit + lv] = np.stack(v[:lv], axis=0)

        return layout


class DatasetNormalization(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    @property
    def bbox_dims(self):
        return 10

    @property
    def feature_size(self):
        return 10

    def __getitem__(self, idx):
        scene = self._dataset[idx]

        valid_abs_index = np.where(scene[:, -1])[0]
        center = np.mean(scene[valid_abs_index, 3:5], axis=0)
        scene_normalize = copy.deepcopy(scene)
        scene_normalize[valid_abs_index, 3:5] -= center

        return scene_normalize


class DatasetEncoder(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    @property
    def bbox_dims(self):
        return 16

    @property
    def feature_size(self):
        return 16

    def __getitem__(self, idx):
        sample_params = {}
        scene = self._dataset[idx]
        scene[:, 2] = 0
        scene[:, :2] = scene[:, :2] / (np.linalg.norm(scene[:, :2], axis=-1, keepdims=True) + 1e-8)

        x_abs = np.zeros((self.num_class * self.furniture_limit, 10))
        x_abs_r = np.zeros((self.num_class * self.furniture_limit, 3, 3))
        
        for c in range(self.num_class):
            offset = c * self.furniture_limit

            mask = np.where(scene[offset: offset + self.furniture_limit, -1] == 1)[0]
            for i, idx in enumerate(mask):
                idx_new = offset + i
                idx_ori = offset + idx

                x_direction = scene[idx_ori, 0:3]
                y_direction = np.zeros((3))
                y_direction[:2] = [-x_direction[1], x_direction[0]]
                z_direction = np.cross(x_direction, y_direction)

                rotate_matrix = np.vstack([x_direction, y_direction, z_direction]).T

                center = scene[idx_ori, 3:6]
                size = scene[idx_ori, 6:9]

                # furniture attributes
                x_abs[idx_new, :9] = np.concatenate((x_direction, center, size))
                # whether furniture exist
                x_abs[idx_new, -1] = 1
                x_abs_r[idx_new] = rotate_matrix

        if not np.all(scene == x_abs):
            print("not equal")
            return None, None

        ''' 
        Calculate X_rel
        '''
        x_rel_dict = dict()
        for i in range(self.num_class * self.furniture_limit):
            x_rel_dict[i] = dict()
            for j in range(self.num_class * self.furniture_limit):

                if x_abs[i, -1] == 1 and x_abs[j, -1] == 1:
                    Ri = x_abs_r[i]
                    ti = x_abs[i][3:6]
                    si = x_abs[i][6:9]
                    Rj = x_abs_r[j]
                    tj = x_abs[j][3:6]
                    sj = x_abs[j][6:9]

                    Rij = Rj.dot(Ri.T)
                    Rij_vec = Rij.T.reshape(-1)

                    tij = tj - ti

                    x_rel_dict[i][j] = np.hstack((Rij_vec[:2], tij, si, sj, 1))  # 2 + 3 + 6 + 1 = 12
        sample_params["x_abs"] = x_abs
        sample_params["x_rel"] = x_rel_dict
        return sample_params


class CachedDataset(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class DatasetDiscrete(CachedDataset):

    def __init__(self, dataset, half_range=6, interval=0.3):
        super().__init__(dataset)
        self.half_range = half_range
        self.interval = interval
        self._dataset = dataset

    @staticmethod
    def angle2class(angle, num_class):
        angle = angle % (2 * np.pi)
        assert (0 <= angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    @staticmethod
    def class2angle(pred_cls, residual, num_class): 
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        return angle

    @staticmethod
    def translation2disc_(tij, size_i, size_j, half_range, interval):
        indicator_ij = (tij > 0)

        dummy = tij.copy()
        for i, item in enumerate(dummy):
            for j, data in enumerate(item):
                if data[0] > 0:
                    data[0] -= (size_i[i][0] + size_j[j][0])
                    np.maximum(data[0], 0)
                else:
                    data[0] += (size_i[i][0] + size_j[j][0])
                    np.minimum(data[0], 0)

        class_id = np.minimum(np.abs(dummy), interval - 1e-8) // interval

        residual = np.abs(dummy) - class_id * interval

        return indicator_ij.astype(np.float32), class_id.astype(np.int), residual

    @staticmethod
    def translation2disc(tij, half_range, interval):

        Iij = (tij > 0)
        class_id = np.minimum(np.abs(tij), interval - 1e-8) // interval
    
        residual = np.abs(tij) - class_id * interval
        return Iij.astype(np.float32), class_id.astype(np.int), residual

    @staticmethod
    def disc2translation(Iij, class_id, residual, half_range, interval):
        sij = (Iij - 0.5) * 2  # 1 or -1
        tij = sij * (class_id * interval + residual)
        return tij

    @staticmethod
    def close(a, b, r=0.10001):
        return np.abs(a - b) <= r

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __getitem__(self, index):
        sample_params = self._dataset[index]
        x_abs = sample_params['x_abs']
        x_rel = sample_params['x_rel']
        n = x_abs.shape[0]
        # The relative size is 12
        x_rel_np = np.zeros((n, n, 12)).astype(np.float32)
        for i in range(n):
            if len(x_rel[i]) > 0:
                for j, v in x_rel[i].items():
                    x_rel_np[i, j, :] = v

        x_abs_sep = np.zeros((n, 16))  # 8 + 1 + 3 + 3 + 1 = 9
        for i, x_abs in enumerate(x_abs):
            if x_abs[-1] > 0.5:
                # angle rad
                angle = np.arctan2(x_abs[1], x_abs[0])
                class_id, residual_angle = DatasetDiscrete.angle2class(angle, num_class=8)
                x_abs_sep[i, int(class_id)] = 1
                x_abs_sep[i, 8:] = np.concatenate(([residual_angle], x_abs[3:]))

        ''' Translation '''
        tn_i_x, tn_class_x, tn_res_x = DatasetDiscrete.translation2disc(
            x_rel_np[:, :, 2:3],
            half_range=self.half_range,
            interval=self.interval
        )

        tn_i_y, tn_class_y, tn_res_y = DatasetDiscrete.translation2disc(
            x_rel_np[:, :, 3:4],
            half_range=self.half_range,
            interval=self.interval
        )

        ''' Rotation '''
        # 0: no relation. 1: parallel. 2: orthogonal
        threshold_r1 = np.cos(np.deg2rad(10))
        threshold_r2 = np.cos(np.deg2rad(80))
        abs_cos = np.abs(x_rel_np[:, :, 0:1])
        rotation_class = (abs_cos > threshold_r1).astype(np.float32) + (abs_cos < threshold_r2).astype(np.float32) * 2

        ''' Size '''
        # thres_s1 = 0.1
        same_size_cond = DatasetDiscrete.close(x_rel_np[:, :, 7:8], x_rel_np[:, :, 10:11]) * \
                         ((DatasetDiscrete.close(x_rel_np[:, :, 5:6], x_rel_np[:, :, 8:9]) *
                           DatasetDiscrete.close(x_rel_np[:, :, 6:7], x_rel_np[:, :, 9:10])) +
                          (DatasetDiscrete.close(x_rel_np[:, :, 5:6], x_rel_np[:, :, 9:10]) *
                           DatasetDiscrete.close(x_rel_np[:, :, 6:7], x_rel_np[:, :, 8:9])))
        same_size = (same_size_cond != 0).astype(np.float32)

        rel_size = np.linalg.norm(x_rel_np[:, :, 8:11], axis=2, keepdims=True) / (
                np.linalg.norm(x_rel_np[:, :, 5:8], axis=2, keepdims=True) + 1e-10)

        x_rel_sep = np.concatenate(
            (tn_i_x, tn_i_y, tn_class_x, tn_class_y, tn_res_x, tn_res_y, x_rel_np[:, :, 4:5],
             rotation_class, same_size, rel_size, x_rel_np[:, :, -1:]), axis=2)

        ret_dict = {
            'index': int(str(sample_params['scene_id']).split('-')[-1]),
            'x_abs': x_abs_sep.astype(np.float32),
            'x_rel': x_rel_sep.astype(np.float32),
        }

        return ret_dict


class FrontFactory:
    @staticmethod
    def encode_dataset(name, dataset):
        if "cached" in name:
            return DatasetDiscrete(CachedDataset(dataset), interval=0.3, half_range=6)
        else:
            box_dataset = BoxDataset(dataset)
            # floorplan = FloorPlanEncoder(box_dataset)
            class_index = ClassIndexEncoder(box_dataset)
            translations = TranslationEncoder(box_dataset)
            sizes = SizeEncoder(box_dataset)
            angles = AngleEncoder(box_dataset)

            dataset = DatasetCollection(
                # floorplan,
                class_index,
                translations,
                sizes,
                angles
            )

            dataset = DatasetNormalization(dataset)
            dataset = DatasetEncoder(dataset)

            return dataset