import copy
import os
import queue
from collections import defaultdict

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from shapely import geometry
from simple_3dviz import Mesh
from skimage import io, transform
from skimage import img_as_ubyte

from synthesis.datasets import THREED_FRONT_FURNITURE


class FloorPlan:
    def __init__(self, file_path):
        self.lamps = None
        self.path = file_path
        self.front_door = None
        self.regions = None
        self.exterior_boundary = None
        self.furniture = None
        self.furniture_list = None
        self.edges = None

        if not os.path.exists(f"{file_path}/room_mask_rotate.png"):
            image = transform.rotate(io.imread(f"{file_path}/room_mask.png", as_gray=True), 90)
            io.imsave(f"{file_path}/room_mask_rotate.png", img_as_ubyte(image))
        self.image = io.imread(f"{file_path}/room_mask_rotate.png", as_gray=True)
        self.h, self.w = self.image.shape

        self.color_maps = [
            # desk
            np.array([39 / 255, 60 / 255, 117 / 255]),
            # nightstand
            np.array([0 / 255, 168 / 255, 255 / 255]),
            # double_bed
            np.array([251 / 255, 197 / 255, 49 / 255]),
            # single_bed
            np.array([76 / 255, 209 / 255, 55 / 255]),
            # kids_bed
            np.array([72 / 255, 126 / 255, 176 / 255]),
            # bookshelf
            np.array([230 / 255, 126 / 255, 34 / 255]),
            # tv_stand
            np.array([52 / 255, 73 / 255, 94 / 255]),
            # wardrobe
            np.array([26 / 255, 188 / 255, 156 / 255]),
            # chair
            np.array([253 / 255, 121 / 255, 168 / 255]),
            # dining_chair
            np.array([99 / 255, 110 / 255, 114 / 255]),
            # chinese_chair
            np.array([255 / 255, 118 / 255, 117 / 255]),
            # armchair
            np.array([253 / 255, 203 / 255, 110 / 255]),
            # dressing_table
            np.array([225 / 255, 112 / 255, 85 / 255]),
            # dressing_chair
            np.array([224 / 255, 86 / 255, 253 / 255]),
            # corner_side_table
            np.array([199 / 255, 236 / 255, 238 / 255]),
            # dining_table
            np.array([0 / 255, 148 / 255, 50 / 255]),
            # coffee_table
            np.array([131 / 255, 52 / 255, 113 / 255]),
            # cabinet
            np.array([120 / 255, 224 / 255, 143 / 255]),
            # shelf
            np.array([196 / 255, 69 / 255, 105 / 255]),
            # stool
            np.array([48 / 255, 57 / 255, 82 / 255]),
            # sofa
            np.array([132 / 255, 129 / 255, 122 / 255]),
            # floor_lamp
            np.array([255 / 255, 184 / 255, 184 / 255]),
            # lamp
            np.array([204 / 255, 174 / 255, 98 / 255]),
        ]
        # self._get_exterior_boundary_vertices()
        try:
            self._get_exterior_boundary()
        except AssertionError as assert_error:
            raise assert_error
        except ValueError as value_error:
            raise value_error
        except UnboundLocalError as unknow_error:
            raise unknow_error

    @property
    def boundary(self):
        return self.image

    @property
    def attrs(self):
        return np.load(f"{self.path}/boxes.npz")

    @property
    def scene_id(self):
        return ''.join(self.attrs["scene_id"].tolist())

    @property
    def scene_type(self):
        return self.attrs["scene_type"]

    @property
    def scene_uid(self):
        return self.attrs["scene_uid"]

    @property
    def original_vertices(self):
        return self.attrs['floor_plan_vertices']

    @property
    def offset(self):
        return self.attrs["floor_plan_centroid"]

    @property
    def original_faces(self):
        return self.attrs["floor_plan_faces"]

    @property
    def original_mesh(self):
        return trimesh.Trimesh(self.original_vertices - self.offset, self.original_faces, process=True)

    @property
    def vertices(self):
        vertices = np.around(self.original_mesh.vertices, decimals=2).tolist()
        edge = self.original_mesh.edges.tolist()

        def mid(a, b, c):
            if c < a and c < b: return 1 if b < a else 0
            r = a if a > b else b
            if r < c:
                return 2
            elif r == a:
                return 0
            else:
                return 1

        invalid_index = set()
        for i in range(0, len(edge), 3):

            x = vertices[edge[i][0]]
            y = vertices[edge[i + 1][0]]
            z = vertices[edge[i + 2][0]]
            # (1 / 2) * (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
            area = (1 / 2) * (x[2] * y[0] + y[2] * z[0] + z[2] * y[0] - x[2] * z[0] - y[2] * x[0] - z[2] * y[0])
            if area == 0:
                if x[0] == y[0] and y[0] == z[0]:
                    invalid_index.add(edge[i + mid(x[2], y[2], z[2])][0])
                elif x[2] == y[2] and y[2] == z[2]:
                    invalid_index.add(edge[i + mid(x[0], y[0], z[0])][0])

        vertices = [vertex for index, vertex in enumerate(vertices) if index not in invalid_index]

        # Remove wrong point
        x_dict = defaultdict(int)
        y_dict = defaultdict(int)
        for i in range(len(vertices)):
            x_dict[str(vertices[i][2])] += 1
            y_dict[str(vertices[i][0])] += 1

        vertices = [
            vertex for vertex in vertices
            if x_dict[str(vertex[2])] % 2 == 0 and y_dict[str(vertex[0])] % 2 == 0
        ]

        return np.around(vertices, decimals=2)

    def _class2angle(self, pred_cls, residual, num_class):
        """
        Inverse function to angle to class
        :param pred_cls: (same_shape)
        :param residual: (same_shape)
        :param num_class: num of class (default: 32)

        :return angle: (same_shape). 0~2pi
        """

        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center
        return angle

    def _get_furniture_list(self, num_class):
        cat2id = dict()
        index = 0
        for d in [THREED_FRONT_FURNITURE]:
            for k, v in d.items():
                if v not in cat2id.keys():
                    cat2id[v] = index
                    index += 1
        self.id2cat = {val: key for key, val in cat2id.items()}
        self.furniture_list = [self.id2cat[key] for key in list(range(num_class))]

    def load_furniture_from_raw_data(self,
                                     data,
                                     num_class=23,
                                     num_each_class=4,
                                     thresholds=0.7,
                                     floor_plan_centroid=None,
                                     with_floor_plan_offset=False):
        if self.furniture_list is None:
            self._get_furniture_list(num_class)

        if with_floor_plan_offset:
            offset = -floor_plan_centroid
        else:
            offset = [0, 0, 0]

        # Ignore Lamps
        furniture = []

        for i in range(num_class - 1):
            for j in range(num_each_class):
                furniture_object_data = data[i * num_each_class + j, :]

                if furniture_object_data[-1] > thresholds:
                    # get class
                    c = i
                    y0, x0, y1, x1, size, z, direction = self.load_single_furniture(
                        object_data=furniture_object_data,
                        offset=offset
                    )
                    furniture.append(
                        [y0, x0, y1, x1, z, size[0], size[1], size[2], direction[0], direction[1], direction[2], c])

        lamps = []
        for j in range(num_each_class):
            furniture_object_data = data[(num_class - 1) * num_each_class + j, :]

            if furniture_object_data[-1] > thresholds:
                # get class
                y0, x0, y1, x1, size, z, direction = self.load_single_furniture(
                    object_data=furniture_object_data,
                    offset=offset
                )

                lamps.append(
                    [y0, x0, y1, x1, z, size[0], size[1], size[2], direction[0], direction[1], direction[2], 22])
        if len(furniture) <= 0:
            return False
        
        furniture = np.array(furniture, dtype=float)
        order_furniture = furniture[np.lexsort(-furniture.T)]
        self.furniture = np.array(order_furniture, dtype=float)
        self.lamps = np.array(lamps, dtype=float)

        self._get_edges()
        return True

    def load_furniture_from_data(self,
                                 data,
                                 num_class=23,
                                 num_each_class=4,
                                 thresholds=0.5,
                                 with_floor_plan_offset=False):
        """
        load furniture from matrix data
        """
        if self.furniture_list is None:
            self._get_furniture_list(num_class)

        if with_floor_plan_offset:
            offset = -self.attrs["floor_plan_centroid"]
        else:
            offset = [0, 0, 0]

        # Ignore Lamps
        furniture = []

        for i in range(num_class - 1):
            for j in range(num_each_class):
                furniture_object_data = data[i * num_each_class + j, :]

                if furniture_object_data[-1] > thresholds:
                    # get class
                    c = i
                    y0, x0, y1, x1, size, z, direction = self.load_single_furniture(
                        object_data=furniture_object_data,
                        offset=offset
                    )
                    furniture.append(
                        [y0, x0, y1, x1, z, size[0], size[1], size[2], direction[0], direction[1], direction[2], c])

        lamps = []
        for j in range(num_each_class):
            furniture_object_data = data[(num_class - 1) * num_each_class + j, :]

            if furniture_object_data[-1] > thresholds:
                # get class
                y0, x0, y1, x1, size, z, direction = self.load_single_furniture(
                    object_data=furniture_object_data,
                    offset=offset
                )

                lamps.append(
                    [y0, x0, y1, x1, z, size[0], size[1], size[2], direction[0], direction[1], direction[2], 22])
        if len(furniture) <= 0:
            return False
        furniture = np.array(furniture, dtype=float)
        order_furniture = furniture[np.lexsort(-furniture.T)]
        self.furniture = np.array(order_furniture, dtype=float)
        self.lamps = np.array(lamps, dtype=float)

        self._get_edges()
        return True

    def load_single_furniture(self, object_data, offset):
        if len(object_data) == 16:
            center = object_data[9:12]
            size = object_data[12:15]
            angle_recon = self._class2angle(
                np.argmax(object_data[:8]),
                object_data[8],
                num_class=8
            )

            dir_1 = np.array([np.cos(angle_recon), np.sin(angle_recon), 0])
            dir_2 = np.zeros(3)
            dir_2[:2] = [-dir_1[1], dir_1[0]]
            dir_3 = np.cross(dir_1, dir_2)

            offset = copy.deepcopy(offset)
            offset = [offset[0], offset[2], 0]
        elif len(object_data) == 10:
            center = object_data[3:6]
            size = object_data[6:9]
            angle = np.arctan2(object_data[1], object_data[0])

            dir_1 = np.array([np.cos(angle), np.sin(angle), 0])
            dir_2 = np.zeros(3)
            dir_2[:2] = [-dir_1[1], dir_1[0]]
            dir_3 = np.cross(dir_1, dir_2)

            offset = copy.deepcopy(offset)
            offset = [offset[0], offset[2], 0]

        cornerpoints = np.zeros([8, 3])
        d1 = 0.5 * size[1] * dir_1
        d2 = 0.5 * size[0] * dir_2
        d3 = 0.5 * size[2] * dir_3

        cornerpoints[0][:] = (center - d1 - d2 - d3) + offset
        cornerpoints[1][:] = (center - d1 + d2 - d3) + offset
        cornerpoints[2][:] = (center + d1 - d2 - d3) + offset
        cornerpoints[3][:] = (center + d1 + d2 - d3) + offset
        cornerpoints[4][:] = (center - d1 - d2 + d3) + offset
        cornerpoints[5][:] = (center - d1 + d2 + d3) + offset
        cornerpoints[6][:] = (center + d1 - d2 + d3) + offset
        cornerpoints[7][:] = (center + d1 + d2 + d3) + offset

        min_x = np.min(cornerpoints[..., 0])
        max_x = np.max(cornerpoints[..., 0])
        min_y = np.min(cornerpoints[..., 1])
        max_y = np.max(cornerpoints[..., 1])
        z = (np.max(cornerpoints[..., 2]) + np.min(cornerpoints[..., 2])) / 2

        return max_y, max_x, min_y, min_x, size, z, np.around(dir_1, decimals=2)

    def _get_linear_equation(self, p1x, p1y, p2x, p2y):
        sign = 1
        a = p2y - p1y
        if a < 0:
            sign = -1
            a = sign * a
        b = sign * (p1x - p2x)
        c = sign * (p1y * p2x - p1x * p2y)
        return [a, b, c]

    def _get_edges(self):
        """
        relative of each two furniture
        """
        assert self.furniture is not None
        edges = []

        for u in range(len(self.furniture)):
            uy0, ux0, uy1, ux1, z1, _, _, _, _, _, _, _ = self.furniture[u]
            line1 = self._get_linear_equation(ux1, uy0, ux0, uy1)
            line2 = self._get_linear_equation(ux0, uy0, ux1, uy1)
            uy, ux = (uy0 + uy1) / 2, (ux0 + ux1) / 2
            for v in range(u + 1, len(self.furniture)):

                # right/above point and left/bottom point
                vy0, vx0, vy1, vx1, z2, _, _, _, _, _, _, _ = self.furniture[v]
                vy, vx = (vy0 + vy1) / 2, (vx0 + vx1) / 2
                # 'left-above'
                if vx0 <= ux1 and uy0 <= vy1:
                    relation = 0
                # 'right-above'
                elif ux0 <= vx1 and uy0 <= vy1:
                    relation = 8
                # 'right-below'
                elif ux0 <= vx1 and uy1 >= vy0:
                    relation = 9
                # 'left-below'
                elif vx0 <= ux1 and uy1 >= vy0:
                    relation = 1
                # 'above'
                elif uy < vy and line1[0] * vx + line1[1] * vy + line1[2] >= 0 and line2[0] * vx + line2[1] * vy + \
                        line2[2] <= 0:
                    relation = 3
                # 'right-of'
                elif vx > ux and line1[0] * vx + line1[1] * vy + line1[2] >= 0 and line2[0] * vx + line2[1] * vy + \
                        line2[2] >= 0:
                    relation = 7
                # 'left-of'
                elif vx < ux and line1[0] * vx + line1[1] * vy + line1[2] <= 0 and line2[0] * vx + line2[1] * vy + \
                        line2[2] <= 0:
                    relation = 2
                # 'below'
                elif vy < uy and line1[0] * vx + line1[1] * vy + line1[2] <= 0 and line2[0] * vx + line2[1] * vy + \
                        line2[2] >= 0:
                    relation = 6
                # 'inside'
                else:
                    relation = 10
                    print(f"no relation")
                    fig = plt.figure(figsize=(15, 15))
                    ax = fig.add_subplot(111)
                    maxr, maxc, minr, minc = self.furniture[u][0:4]
                    x = (minc, maxc, maxc, minc, minc)
                    y = (minr, minr, maxr, maxr, minr)
                    ax.plot(x, y, '-b', linewidth=1.5)

                    maxr, maxc, minr, minc = self.furniture[v][0:4]
                    x = (minc, maxc, maxc, minc, minc)
                    y = (minr, minr, maxr, maxr, minr)
                    ax.plot(x, y, '-k', linewidth=1.5)

                    plt.tight_layout()
                    plt.savefig("error.png")
                    plt.close()
                edges.append([u, v, relation])

        self.edges = np.array(edges, dtype=int)

    def _get_front_door(self):
        raise NotImplementedError
    
    def _point_sort(self, l):
        if len(l) <= 1:
            return l
        mid = l[0]
        low = [item for item in l if item < mid]
        high = [item for item in l if item > mid]
        return self._point_sort(low) + [mid] + self._point_sort(high)

    def _dist2(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2

    def _get_exterior_boundary(self):
        real_points = np.array(self.vertices.tolist())
        exterior_boundary = []

        min_h, max_h = np.where(np.any(self.boundary, axis=1))[0][[0, -1]]
        min_w, max_w = np.where(np.any(self.boundary, axis=0))[0][[0, -1]]
        min_h = max(min_h - 10, 0)
        min_w = max(min_w - 10, 0)
        max_h = min(max_h + 10, self.h)
        max_w = min(max_w + 10, self.w)

        # src: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
        # search direction:0(right)/1(down)/2(left)/3(up)
        # find the left-top point
        flag = False
        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                if self.boundary[h, w] == 255:
                    exterior_boundary.append((h, w, 0))
                    flag = True
                    break
            if flag:
                break
        try:
            while flag:
                new_point = None
                if exterior_boundary[-1][2] == 0:
                    for w in range(exterior_boundary[-1][1] + 1, max_w):
                        corner_sum = 0
                        if self.boundary[exterior_boundary[-1][0], w] == 255:
                            corner_sum += 1
                        if self.boundary[exterior_boundary[-1][0] - 1, w] == 255:
                            corner_sum += 1
                        if self.boundary[exterior_boundary[-1][0], w - 1] == 255:
                            corner_sum += 1
                        if self.boundary[exterior_boundary[-1][0] - 1, w - 1] == 255:
                            corner_sum += 1
                        if corner_sum == 1:
                            new_point = (exterior_boundary[-1][0], w, 1)
                            break
                        if corner_sum == 3:
                            new_point = (exterior_boundary[-1][0], w, 3)
                            break

                if exterior_boundary[-1][2] == 1:
                    for h in range(exterior_boundary[-1][0] + 1, max_h):
                        corner_sum = 0
                        if self.boundary[h, exterior_boundary[-1][1]] == 255:
                            corner_sum += 1
                        if self.boundary[h - 1, exterior_boundary[-1][1]] == 255:
                            corner_sum += 1
                        if self.boundary[h, exterior_boundary[-1][1] - 1] == 255:
                            corner_sum += 1
                        if self.boundary[h - 1, exterior_boundary[-1][1] - 1] == 255:
                            corner_sum += 1
                        if corner_sum == 1:
                            new_point = (h, exterior_boundary[-1][1], 2)
                            break
                        if corner_sum == 3:
                            new_point = (h, exterior_boundary[-1][1], 0)
                            break

                if exterior_boundary[-1][2] == 2:
                    for w in range(exterior_boundary[-1][1] - 1, min_w, -1):
                        corner_sum = 0
                        if self.boundary[exterior_boundary[-1][0], w] == 255:
                            corner_sum += 1
                        if self.boundary[exterior_boundary[-1][0] - 1, w] == 255:
                            corner_sum += 1
                        if self.boundary[exterior_boundary[-1][0], w - 1] == 255:
                            corner_sum += 1
                        if self.boundary[exterior_boundary[-1][0] - 1, w - 1] == 255:
                            corner_sum += 1
                        if corner_sum == 1:
                            new_point = (exterior_boundary[-1][0], w, 3)
                            break
                        if corner_sum == 3:
                            new_point = (exterior_boundary[-1][0], w, 1)
                            break

                if exterior_boundary[-1][2] == 3:
                    for h in range(exterior_boundary[-1][0] - 1, min_h, -1):
                        corner_sum = 0
                        if self.boundary[h, exterior_boundary[-1][1]] == 255:
                            corner_sum += 1
                        if self.boundary[h - 1, exterior_boundary[-1][1]] == 255:
                            corner_sum += 1
                        if self.boundary[h, exterior_boundary[-1][1] - 1] == 255:
                            corner_sum += 1
                        if self.boundary[h - 1, exterior_boundary[-1][1] - 1] == 255:
                            corner_sum += 1
                        if corner_sum == 1:
                            new_point = (h, exterior_boundary[-1][1], 0)
                            break
                        if corner_sum == 3:
                            new_point = (h, exterior_boundary[-1][1], 2)
                            break
                assert new_point != None
                if new_point != exterior_boundary[0]:
                    exterior_boundary.append(new_point)
                else:
                    flag = False
        except ValueError:
            raise
        except UnboundLocalError:
            raise

        # exterior_boundary = [[r, c, d, 0] for r, c, d in exterior_boundary]
        assert len(exterior_boundary) != 0
        self.exterior_boundary = [
            [(500 - exterior_boundary[0][0]), 0, (exterior_boundary[0][1] - 500), 0]
        ]

        for i in range(1, len(exterior_boundary)):
            r = (500 - exterior_boundary[i][0])
            c = (exterior_boundary[i][1] - 500)

            pre_r = self.exterior_boundary[-1][0]
            pre_c = self.exterior_boundary[-1][2]

            if pre_c > c and pre_r == r:
                self.exterior_boundary.append([r, 0, c, 3])
            elif pre_c < c and pre_r == r:
                self.exterior_boundary.append([r, 0, c, 1])
            elif pre_r > r and pre_c == c:
                self.exterior_boundary.append([r, 0, c, 2])
            elif pre_r < r and pre_c == c:
                self.exterior_boundary.append([r, 0, c, 0])
            elif pre_c < c and pre_r < r:
                self.exterior_boundary.append([r, 0, c, 4])
            elif pre_c < c and pre_r > r:
                self.exterior_boundary.append([r, 0, c, 5])
            elif pre_c > c and pre_r < r:
                self.exterior_boundary.append([r, 0, c, 6])
            elif pre_c > c and pre_r > r:
                self.exterior_boundary.append([r, 0, c, 7])

        self.exterior_boundary = np.array(self.exterior_boundary, dtype=float)
        assert len(real_points) == len(self.exterior_boundary)

        real_x_order = np.array(self._point_sort(real_points[..., 2]))
        real_y_order = np.array(self._point_sort(real_points[..., 0]))
        image_x_order = np.array(self._point_sort(self.exterior_boundary[..., 2]))
        image_y_order = np.array(self._point_sort(self.exterior_boundary[..., 0]))

        map_real = np.zeros([len(real_x_order), len(real_y_order)])
        for k in range(len(real_points)):
            i = np.where(real_x_order == real_points[k][2])[0].tolist()[0]
            j = np.where(real_y_order == real_points[k][0])[0].tolist()[0]
            map_real[i][j] = k

        real_boundary = []
        for k in range(len(self.exterior_boundary)):
            i = np.where(image_x_order == self.exterior_boundary[k][2])[0].tolist()[0]
            j = np.where(image_y_order == self.exterior_boundary[k][0])[0].tolist()[0]
            real_index = int(map_real[i][j])
            real_boundary.append(
                [real_points[real_index][2], 0, real_points[real_index][0], self.exterior_boundary[k][3]])
        self.exterior_boundary = np.array(real_boundary, dtype=float)

    def to_dict(self, floor_plan_only=True, xyxy=True):
        return {
            'name': self.scene_id,
            'type': np.array([], dtype=int) if floor_plan_only else self.furniture[:, -1].astype(int) if self.furniture is not None else np.array([], dtype=int),
            'boundary': self.exterior_boundary[:, [2, 1, 0, 3]].astype(float),
            'boxes': np.array([], dtype=int) if floor_plan_only else self.furniture.astype(float) if self.furniture is not None else np.array([], dtype=int),
            'lamps': np.array([], dtype=int) if floor_plan_only else self.lamps.astype(float) if self.lamps is not None else np.array([], dtype=int),
            'edges': np.array([], dtype=int) if floor_plan_only else self.edges.astype(int) if self.edges is not None else np.array([], dtype=int),
            'colors': np.array([], dtype=int) if floor_plan_only else np.array(self.color_maps).astype(float)
        }
