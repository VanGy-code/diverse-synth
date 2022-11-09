import os
from collections import Counter
from functools import lru_cache, cached_property, reduce

import numpy as np
from simple_3dviz import Mesh

class BaseScene(object):
    def __init__(self, scene_id, scene_type, bboxes):
        self.bboxes = bboxes
        self.scene_id = scene_id
        self.scene_type = scene_type

    def __str__(self):
        return "Scene: {} of type: {} contains {} bboxes".format(
            self.scene_id, self.scene_type, self.num_objects
        )

    @property
    def num_objects(self):
        return len(self.bboxes)


class Room(BaseScene):
    def __init__(self, scene_id, scene_type, bboxes, extras, json_path,
                 path_to_room_masks_dir=None):
        super().__init__(scene_id, scene_type, bboxes)
        self.json_path = json_path
        self.extras = extras

        self.uid = "_".join([self.json_path, scene_id])
        self.path_to_room_masks_dir = path_to_room_masks_dir
        if path_to_room_masks_dir is not None:
            self.path_to_room_mask = os.path.join(
                self.path_to_room_masks_dir, self.uid, "room_mask.png"
            )
        else:
            self.path_to_room_mask = None

    @property
    def floor(self):
        return [ei for ei in self.extras if ei.model_type == "Floor"][0]

    @property
    @lru_cache(maxsize=512)
    def bbox(self):
        corners = np.empty((0, 3))
        for f in self.bboxes:
            corners = np.vstack([corners, f.corners()])
        return np.min(corners, axis=0), np.max(corners, axis=0)

    @cached_property
    def bboxes_centroid(self):
        a, b = self.bbox
        return (a + b) / 2

    @property
    def furniture_in_room(self):
        return [f.label for f in self.bboxes]

    @property
    def floor_plan(self):
        def cat_mesh(m1, m2):
            v1, f1 = m1
            v2, f2 = m2
            v = np.vstack([v1, v2])
            f = np.vstack([f1, f2 + len(v1)])
            return v, f
        # Compute the full floor plan
        vertices, faces = reduce(
            cat_mesh,
            ((ei.xyz, ei.faces) for ei in self.extras if ei.model_type == "Floor")
        )

        return np.copy(vertices), np.copy(faces)

    @cached_property
    def floor_plan_bbox(self):
        vertices, faces = self.floor_plan
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    @property
    def walls(self):
        walls_list = []
        for ei in self.extras:
            if ei.model_type == "WallInner":
                walls_list.append(ei)
        return walls_list

    @cached_property
    def floor_plan_centroid(self):
        a, b = self.floor_plan_bbox
        return (a + b) / 2

    @cached_property
    def centroid(self, offeset=None):
        return self.floor_plan_centroid

    @property
    def count_furniture_in_room(self):
        return Counter(self.furniture_in_room)

    def floor_plan_renderable(self, with_objects_offset=False,
                              with_floor_plan_offset=True, with_door=False, color=(1.0, 1.0, 1.0, 1.0)):
        vertices, faces = self.floor_plan

        if with_objects_offset:
            offset = -self.bboxes_centroid
        elif with_floor_plan_offset:
            offset = -self.floor_plan_centroid
        else:
            offset = [[0, 0, 0]]

        renderables = [
            Mesh.from_faces(vertices + offset, faces, color)
        ]

        if with_door:
            for door in self.entrance:
                offset = -self.centroid
                renderables += [door.bbox_renderable(offset=offset)]

        return renderables

    def furniture_renderables(
            self,
            colors=(0.5, 0.5, 0.5),
            with_bbox_corners=False,
            with_origin=False,
            with_bboxes=False,
            with_objects_offset=False,
            with_floor_plan_offset=False,
            with_floor_plan=False,
            with_texture=False
    ):
        if with_objects_offset:
            offset = -self.bboxes_centroid
        elif with_floor_plan_offset:
            offset = -self.floor_plan_centroid
        else:
            offset = [[0, 0, 0]]

        renderables = [
            f.mesh_renderable(
                colors=colors, offset=offset, with_texture=with_texture
            )
            for f in self.bboxes
        ]
        if with_origin:
            renderables += [f.origin_renderable(offset) for f in self.bboxes]
            renderables += [door.origin_renderable(offset) for door in self.entrance]
        if with_bbox_corners:
            renderables += [f.bbox_corners_renderable(offset=offset) for f in self.bboxes]
            renderables += [door.bbox_corners_renderable(offset) for door in self.entrance]
        if with_bboxes:
            renderables += [f.bbox_renderable(offset=offset) for f in self.bboxes]
            renderables += [door.bbox_renderable(offset=offset) for door in self.entrance]
        if with_floor_plan:
            vertices, faces = self.floor_plan
            vertices = vertices + offset
            renderables += [
                Mesh.from_faces(vertices, faces, colors=(0.8, 0.8, 0.8, 0.6))
            ]
        return renderables


class CachedRoom(object):
    def __init__(
            self,
            scene_id,
            floor_plan_mask,
            floor_plan_vertices,
            floor_plan_faces,
            floor_plan_centroid,
            class_labels,
            translations,
            sizes,
            angles,
            image_path,
    ):
        self.scene_id = scene_id
        self.floor_plan_mask = floor_plan_mask
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.floor_plan_mask[:, :, None]