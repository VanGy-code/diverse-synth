import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import trimesh
from simple_3dviz import Mesh, TexturedMesh, Lines, Spherecloud
from simple_3dviz.io import read_mesh_file
from simple_3dviz.renderables.textured_mesh import Material


def rotation_matrix(axis, theta):
    """Axis-angle rotation matrix from 3D-Front-Toolbox."""
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class Furniture:

    def __init__(self, super_category, category, style, theme, material):
        self.super_category = super_category
        self.category = category
        self.style = style
        self.theme = theme
        self.material = material

    @property
    def label(self):
        return self.category


@dataclass
class Asset:
    super_category: str
    category: str
    style: str
    theme: str
    material: str

    @property
    def label(self):
        return self.category


class BaseFutureModel(object):
    def __init__(self, model_uid, model_jid, instance_id, position, rotation, scale):
        self.model_uid = model_uid
        self.model_jid = model_jid
        self.instance_id = instance_id
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def _transform(self, vertices):
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2
        vertices = vertices * self.scale
        if np.sum(axis) != 0 and not np.isnan(theta):
            R = rotation_matrix(axis, theta)
            vertices = vertices.dot(R.T)
        vertices += self.position

        return vertices

    def mesh_renderable(
            self,
            colors=(0.5, 0.5, 0.5, 1.0),
            offset=[[0, 0, 0]],
            with_texture=False
    ):
        if not with_texture:
            m = self.raw_model_transformed(offset)
            return Mesh.from_faces(m.vertices, m.original_faces, colors=colors)
        else:
            model_path = self.raw_model_path
            try:
                m = TexturedMesh.from_file(model_path)
            except:
                try:
                    texture_path = self.texture_image_path
                    mesh_info = read_mesh_file(model_path)
                    vertices = mesh_info.vertices
                    normals = mesh_info.normals
                    uv = mesh_info.uv
                    material = Material.with_texture_image(texture_path)
                    m = TexturedMesh(vertices, normals, uv, material)
                except Exception:
                    print("Failed loading texture info.")
                    m = Mesh.from_file(model_path)
            m.scale(self.scale)
            theta = self.z_angle
            R = np.zeros((3, 3))
            R[0, 0] = np.cos(theta)
            R[0, 2] = -np.sin(theta)
            R[2, 0] = np.sin(theta)
            R[2, 2] = np.cos(theta)
            R[1, 1] = 1.


            m.affine_transform(R=R, t=self.position)
            m.affine_transform(t=offset)
            return m


class FutureModel(BaseFutureModel):
    def __init__(
            self,
            model_uid,
            model_jid,
            instance_id,
            model_info,
            position,
            rotation,
            scale,
            path_to_models
    ):
        super().__init__(model_uid, model_jid, instance_id, position, rotation, scale)
        self.model_info = model_info
        self.path_to_models = path_to_models
        self._label = None

    @property
    def raw_model_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model.obj"
        )

    @property
    def texture_image_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "texture.png"
        )

    @property
    def label(self):
        if self._label is None:
            self._label = self.model_info.category
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "bbox_vertices.npy"
        )

    @cached_property
    def vertices(self):
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except Exception:
            bbox_vertices = np.array(self.load_raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        return bbox_vertices

    @cached_property
    def center(self):
        return (np.max(self.vertices, axis=0) + np.min(self.vertices, axis=0)) / 2

    @cached_property
    def half_size(self):
        return (np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)) / 2

    @cached_property
    def size(self):
        """
        Compute model size
        """
        corners = self.corners()
        return np.array([
            # half x
            np.sqrt(np.sum((corners[4] - corners[0]) ** 2)) / 2,
            # half y
            np.sqrt(np.sum((corners[2] - corners[0]) ** 2)) / 2,
            # half z
            np.sqrt(np.sum((corners[1] - corners[0]) ** 2)) / 2
        ])

    @cached_property
    def bottom_size(self):
        return self.size * [1, 2, 1]

    @cached_property
    def z_angle(self):
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2

        if np.sum(axis) == 0 or np.isnan(theta):
            return 0

        assert np.dot(axis, [1, 0, 1]) == 0
        assert 0 <= theta <= 2 * np.pi

        if theta >= np.pi:
            theta = theta - 2 * np.pi

        return np.sign(axis[1]) * theta

    def load_raw_model(self):
        try:
            return trimesh.load(
                self.raw_model_path,
                force="mesh",
            )
        except Exception:
            print("Loading model failed", flush=True)
            raise

    def raw_model_transformed(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        model = self.load_raw_model()
        faces = np.array(model.original_faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

    def corners(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except Exception:
            bbox_vertices = np.array(self.load_raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        return c + offset

    def centroid(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        return self.corners(offset).mean(axis=0)

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def bottom_center(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1] - size[1], centroid[2]])

    def direction(self, offset=None):
        corners = self.corners(offset)
        face_z_center = np.mean(corners[1:9:2], axis=0)
        # x,y,z direction
        direction = face_z_center - self.centroid(offset)
        # L2 Norm
        direction = direction / np.linalg.norm(direction)

        return direction

    def gen_box_from_params(self, offset=None):
        direction = self.direction(offset)
        centroid = self.centroid(offset)
        scale = self.size * self.scale * 2

        p = np.concatenate([
            [direction[2], direction[0], direction[1]],
            [centroid[2], centroid[0], centroid[1]],
            [scale[0], scale[2], scale[1]]
        ])

        # bbox definition
        dir_1 = np.zeros(3)
        dir_1[:2] = p[:2]
        dir_1 = dir_1 / np.linalg.norm(dir_1)
        dir_2 = np.zeros(3)
        dir_2[:2] = [-dir_1[1], dir_1[0]]
        dir_3 = np.cross(dir_1, dir_2)

        center = p[3:6]
        size = p[6:9]

        corner_points = np.zeros([8, 3])
        d1 = 0.5 * size[1] * dir_1
        d2 = 0.5 * size[0] * dir_2
        d3 = 0.5 * size[2] * dir_3
        # d3 = 0
        corner_points[0][:] = center - d1 - d2 - d3
        corner_points[1][:] = center - d1 + d2 - d3
        corner_points[2][:] = center + d1 - d2 - d3
        corner_points[3][:] = center + d1 + d2 - d3
        corner_points[4][:] = center - d1 - d2 + d3
        corner_points[5][:] = center - d1 + d2 + d3
        corner_points[6][:] = center + d1 - d2 + d3
        corner_points[7][:] = center + d1 + d2 + d3

        return corner_points

    def origin_renderable(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        corners = self.corners(offset)
        return Lines(
            [
                corners[0], corners[4],
                corners[0], corners[2],
                corners[0], corners[1]
            ],
            colors=np.array([
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ]),
            width=0.02
        )

    def bbox_corners_renderable(
            self, sizes=0.1, colors=(1, 0, 0), offset=[[0, 0, 0]]
    ):
        return Spherecloud(self.corners(offset), sizes=sizes, colors=colors)

    def bbox_renderable(
            self, colors=(0.00392157, 0., 0.40392157, 1.), offset=[[0, 0, 0]]
    ):
        alpha = np.array(self.size)[None]
        epsilon = np.ones((1, 2)) * 0.1
        translation = np.array(self.centroid(offset))[None]
        R = np.zeros((1, 3, 3))
        theta = np.array(self.z_angle)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 2] = -np.sin(theta)
        R[:, 2, 0] = np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        R[:, 1, 1] = 1.

        return Mesh.from_superquadrics(alpha, epsilon, translation, R, colors)

    def copy_from_other_model(self, other_model):
        model = FutureModel(
            model_uid=other_model.model_uid,
            model_jid=other_model.model_jid,
            instance_id=other_model.instance_id,
            model_info=other_model.model_info,
            position=self.position,
            rotation=self.rotation,
            scale=other_model.scale,
            path_to_models=self.path_to_models
        )
        model.label = self.label
        return model


class Door(BaseFutureModel):
    def __init__(self,
                 model_uid,
                 model_jid,
                 instance_id,
                 label,
                 door_type,
                 position,
                 rotation,
                 scale,
                 path_to_models
                 ):
        super().__init__(model_uid, model_jid, instance_id, position, rotation, scale)
        self.path_to_models = path_to_models
        self.label = label
        self.door_type = door_type
        self._label = None

    @property
    def raw_model_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model.obj"
        )

    @property
    def texture_image_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "texture.png"
        )

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "bbox_vertices.npy"
        )

    def load_raw_model(self):
        try:
            return trimesh.load(
                self.raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,
                skip_texture=True
            )
        except Exception:
            print("Loading model failed", flush=True)
            raise

    @cached_property
    def vertices(self):
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except Exception:
            bbox_vertices = np.array(self.load_raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        return bbox_vertices

    @cached_property
    def center(self):
        return (np.max(self.vertices, axis=0) + np.min(self.vertices, axis=0)) / 2

    @cached_property
    def half_size(self):
        return (np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)) / 2

    @cached_property
    def size(self):
        """
        Compute model size
        """
        corners = self.corners()
        return np.array([
            # half x
            np.sqrt(np.sum((corners[4] - corners[0]) ** 2)) / 2,
            # half y
            np.sqrt(np.sum((corners[2] - corners[0]) ** 2)) / 2,
            # half z
            np.sqrt(np.sum((corners[1] - corners[0]) ** 2)) / 2
        ])

    @cached_property
    def bottom_size(self):
        return self.size * [1, 2, 1]

    @cached_property
    def z_angle(self):
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2

        if np.sum(axis) == 0 or np.isnan(theta):
            return 0

        assert np.dot(axis, [1, 0, 1]) == 0
        assert 0 <= theta <= 2 * np.pi

        if theta >= np.pi:
            theta = theta - 2 * np.pi

        return np.sign(axis[1]) * theta

    def raw_model_transformed(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        vertices = self._transform(np.array(self.xyz)) + offset
        faces = np.array(self.faces)
        return trimesh.Trimesh(vertices, faces)

    def corners(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except Exception:
            bbox_vertices = np.array(self.load_raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        return c + offset

    def centroid(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        return self.corners(offset).mean(axis=0)

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def bottom_center(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1] - size[1], centroid[2]])

    def direction(self, offset=None):
        corners = self.corners(offset)
        face_z_center = np.mean(corners[1:9:2], axis=0)
        # x,y,z direction
        direction = face_z_center - self.centroid(offset)
        # L2 Norm
        direction = direction / np.linalg.norm(direction)

        return direction

    def gen_box_from_params(self, offset=None):
        direction = self.direction(offset)
        centroid = self.centroid(offset)
        scale = self.size * self.scale * 2

        p = np.concatenate([
            [direction[2], direction[0], direction[1]],
            [centroid[2], centroid[0], centroid[1]],
            [scale[0], scale[2], scale[1]]
        ])

        # bbox definition
        dir_1 = np.zeros(3)
        dir_1[:2] = p[:2]
        dir_1 = dir_1 / np.linalg.norm(dir_1)
        dir_2 = np.zeros(3)
        dir_2[:2] = [-dir_1[1], dir_1[0]]
        dir_3 = np.cross(dir_1, dir_2)

        center = p[3:6]
        size = p[6:9]

        corner_points = np.zeros([8, 3])
        d1 = 0.5 * size[1] * dir_1
        d2 = 0.5 * size[0] * dir_2
        d3 = 0.5 * size[2] * dir_3
        # d3 = 0
        corner_points[0][:] = center - d1 - d2 - d3
        corner_points[1][:] = center - d1 + d2 - d3
        corner_points[2][:] = center + d1 - d2 - d3
        corner_points[3][:] = center + d1 + d2 - d3
        corner_points[4][:] = center - d1 - d2 + d3
        corner_points[5][:] = center - d1 + d2 + d3
        corner_points[6][:] = center + d1 - d2 + d3
        corner_points[7][:] = center + d1 + d2 + d3

        return corner_points

    def origin_renderable(self, offset=None):
        if offset is None:
            offset = [[0, 0, 0]]
        corners = self.corners(offset)
        return Lines(
            [
                corners[0], corners[4],
                corners[0], corners[2],
                corners[0], corners[1]
            ],
            colors=np.array([
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ]),
            width=0.02
        )

    def bbox_corners_renderable(
            self, sizes=0.1, colors=(1, 0, 0), offset=[[0, 0, 0]]
    ):
        return Spherecloud(self.corners(offset), sizes=sizes, colors=colors)

    def bbox_renderable(
            self, colors=(1.0, 0.0, 0.0, 1.0), offset=[[0, 0, 0]]
    ):
        alpha = np.array(self.size)[None]
        epsilon = np.ones((1, 2)) * 0.1
        translation = np.array(self.centroid(offset))[None]
        R = np.zeros((1, 3, 3))
        theta = np.array(self.z_angle)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 2] = -np.sin(theta)
        R[:, 2, 0] = np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        R[:, 1, 1] = 1.

        return Mesh.from_superquadrics(alpha, epsilon, translation, R, colors)


class FutureExtra(BaseFutureModel):
    def __init__(
            self,
            model_uid,
            model_jid,
            instance_id,
            xyz,
            faces,
            model_type,
            position,
            rotation,
            scale
    ):
        super().__init__(model_uid, model_jid, instance_id, position, rotation, scale)
        self.xyz = xyz
        self.faces = faces
        self.model_type = model_type