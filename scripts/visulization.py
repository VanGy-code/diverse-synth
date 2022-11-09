import io
import os

import PIL
import numpy as np
import torch
import trimesh
import imageio

import matplotlib
from matplotlib import pyplot as plt, animation
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors

from pyrr import Matrix44

from shapely import geometry

from simple_3dviz import Mesh, Scene, TexturedMesh
from simple_3dviz.io import read_mesh_file
from simple_3dviz.renderables.textured_mesh import Material
from simple_3dviz.utils import save_frame

from synthesis.datasets import THREED_FRONT_FURNITURE

def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color] * len(renderables)
    else:
        color = [None] * len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


class FloorPlanRenderer:
    def __init__(self):
        self.id2cat = None
        self.cat2id = None

        self._get_id2cat()

    def _get_id2cat(self):
        cat2id = dict()
        index = 0
        for d in [THREED_FRONT_FURNITURE]:
            for k, v in d.items():
                if v not in cat2id.keys():
                    cat2id[v] = index
                    index += 1
        self.cat2id = cat2id
        self.id2cat = {val: key for key, val in cat2id.items()}

    @staticmethod
    def set_axes_equal(ax):

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_doors(self, doors, ax=None):
        raise NotImplementedError

    def plot_windows(self, windows, ax=None):
        raise NotImplementedError

    def plot_2d_boundary(self, boundary, ax=None):
        for i in range(len(boundary)):
            if i < len(boundary) - 1:
                p_0 = [boundary[i][2], boundary[i + 1][2]]
                p_1 = [boundary[i][0], boundary[i + 1][0]]
                # x.append(boundary[i][2])
                # y.append(boundary[i][0])
            else:
                p_0 = [boundary[i][2], boundary[0][2]]
                p_1 = [boundary[i][0], boundary[0][0]]
                # x.append(boundary[i][2])
                # y.append(boundary[i][0])

            ax.plot(p_0, p_1, '-k', linewidth=2.5)

    def plot_2d_boundary_mesh(self, boundary, ax=None):
        x = []
        y = []
        for i in range(len(boundary)):
            x.append(boundary[i][2])
            y.append(boundary[i][0])

        x.append(boundary[0][2])
        y.append(boundary[0][0])
        ax.fill(x, y, facecolor='xkcd:gray')


    def plot_2d_label(self, boxes, types, font_size=0, ax=None):
        if font_size != 0:
            for i in range(len(boxes)):
                maxr, maxc, minr, minc = boxes[i][0:4]
                n1 = np.array([boxes[i][-4], boxes[i][-3]])

                cx = (minc + maxc) / 2
                cy = (maxr + minr) / 2
                ax.text(cx, cy, self.id2cat[types[i]], fontsize=font_size, horizontalalignment='center',
                        verticalalignment='center')

                center = np.array([cx, cy])
                line = center + 0.2 * n1
                ax.plot([center[0], line[0]], [center[1], line[1]])

    def plot_2d_furniture_bbox(self, boxes, types, colors, ax=None):
        for i in range(len(boxes)):

            maxr, maxc, minr, minc = boxes[i][0:4]
            x = (minc, maxc, maxc, minc, minc)
            y = (minr, minr, maxr, maxr, minr)
            ax.plot(x, y, c=colors[types[i]], linewidth=1.5)

    def plot_2d_furniture_bbox_mesh(self, boxes, types, colors, ax=None):
        for i in range(len(boxes)):

            maxr, maxc, minr, minc = boxes[i][0:4]
            x = (minc, maxc, maxc, minc, minc)
            y = (minr, minr, maxr, maxr, minr)
            ax.fill(x, y, facecolor=colors[types[i]], alpha=1)


    def plot_2d_fp_mesh(self, boundary=None, boxes=None, types=None, colors=None, font_size=20, filename="2d.png", furniture_only=False,with_text=True):
        fig = plt.figure(figsize=(12, 12), dpi=50)
        ax = fig.add_subplot(111)
        if not furniture_only:
            assert boundary is not None
            self.plot_2d_boundary_mesh(boundary, ax)

        assert boxes is not None and types is not None and colors is not None
        self.plot_2d_furniture_bbox_mesh(boxes, types, colors, ax)

        if with_text:
            self.plot_2d_label(boxes, types, font_size, ax)

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        # plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_2d_fp(self, boundary, boxes, types, colors, font_size=20, filename="2d.png"):
        fig = plt.figure(figsize=(12, 12), dpi=50)
        ax = fig.add_subplot(111)

        self.plot_2d_boundary(boundary, ax)

        self.plot_2d_furniture_bbox(boxes, types, colors, ax)

        self.plot_2d_label(boxes, types, font_size, ax)

        ax.set_aspect('equal', 'datalim')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_3d_boundary(self, boundary, ax):

        y = boundary[..., 0]
        x = boundary[..., 2]
        z = np.zeros(len(boundary)) - 0.01

        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=[0.7, 0.7, 0.7]))

    def plot_3d_furniture_bbox(self, boxes, types, colors, ax=None):
        for i in range(len(boxes)):
            maxr, maxc, minr, minc = boxes[i][0:4]
            h = boxes[i][4]

            x0 = [minc, minc]
            y0 = [maxr, minr]
            z0 = [0, 0]
            ax.plot3D(x0, y0, z0, zdir='z', c=colors[types[i]])

            x1 = [minc, maxc, maxc, minc]
            y1 = [minr, minr, maxr, maxr]
            z1 = [h, h, h, h]
            verts = [list(zip(x1, x1, z1))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=colors[types[i]]))
            ax.plot3D(x1, y1, z1, zdir='z', c=colors[types[i]])

            x2 = (minc, maxc, maxc, minc)
            y2 = (minr, minr, maxr, maxr)
            z2 = (0, 0, 0, 0)
            verts = [list(zip(x2, y2, z2))]
            ax.plot3D(x2, y2, z2, zdir='z', c=colors[types[i]])
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=colors[types[i]]))

            x4 = (maxc, maxc, maxc, maxc)
            y4 = (minr, minr, maxr, maxr)
            z4 = (0, h, h, 0)
            ax.plot3D(x4, y4, z4, zdir='z', c=colors[types[i]])
            verts = [list(zip(x4, y4, z4))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=colors[types[i]]))

            x5 = (minc, minc, minc, minc)
            y5 = (maxr, maxr, minr, minr)
            z5 = (0, h, h, 0)
            ax.plot3D(x5, y5, z5, zdir='z', c=colors[types[i]])
            verts = [list(zip(x5, y5, z5))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=colors[types[i]]))

            x6 = (minc, minc, maxc, maxc)
            y6 = (maxr, maxr, maxr, maxr)
            z6 = (0, h, h, 0)
            verts = [list(zip(x6, y6, z6))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=colors[types[i]]))

            x7 = (minc, minc, maxc, maxc)
            y7 = (minr, minr, minr, minr)
            z7 = (0, h, h, 0)
            verts = [list(zip(x7, y7, z7))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=colors[types[i]]))

    def plot_3d_fp(self, boundary, boxes, types, colors, doors=[], windows=[], filename="3d"):
        fig = plt.figure(figsize=(12, 12), dpi=50)
        ax = fig.gca(projection='3d')
        FloorPlanRenderer.set_axes_equal(ax)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-1, 4)

        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)

        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)

        self.plot_3d_boundary(boundary, ax)

        self.plot_3d_furniture_bbox(boxes, types, colors, ax)

        if len(doors) > 0:
            # plot_door(doors, wall_thickness / 3, ax)
            self.plot_doors(doors, ax)
        if len(windows) > 0:
            # plot_window(windows, wall_thickness / 3, ax)
            self.plot_windows(windows, ax)
        plt.tight_layout()

        plt.savefig(f"{filename}.png", dpi=96)
        plt.close()

        # trainsition = lambda x, N: (1 + np.sin(-0.5 * np.pi + 2 * np.pi * x / (1.0 * N))) / 2.0
        # frames = []
        # for i in range(64):
        #     horiAngle = 100 * trainsition(i, 40)
        #     vertAngle = 40
        #     ax.view_init(vertAngle, horiAngle)
        #     buffer = io.BytesIO()
        #     plt.savefig(buffer, format='png', dpi=96)
        #     buffer.seek(0)
        #     frames.append(imageio.imread(buffer))
        #     buffer.close()
        # plt.close()
        # imageio.mimsave(f"{filename}.gif", frames, fps=10)
