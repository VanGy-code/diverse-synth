import argparse
import json
import logging
import os.path
import random
import sys

import numpy as np
import torch.cuda
import trimesh
from scipy.io import savemat, loadmat
from PIL import Image

from simple_3dviz import TexturedMesh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis.datasets.FutureDataset import FutureDataset
from synthesis.postprocess.boundary2mesh import boundary2mesh
from synthesis.postprocess.box2object import get_furniture_by_params
from scripts.visulization import FloorPlanRenderer
from synthesis.datasets.Common import filter_function
from synthesis.datasets.FRONT import Front
from synthesis.datasets.FloorPlan import FloorPlan
from synthesis.postprocess.processor import Processor


def export_scene(file_path, model_list, boxes, name_pattern=None, is_transfered=False):
    if name_pattern is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(model_list))
        ]
    else:
        names = [
            "{}_{:03d}.obj".format(name_pattern, i) for i in range(len(model_list))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(model_list))
    ]

    for i, m in enumerate(model_list):
        raw_model = trimesh.load(m.raw_model_path, force="mesh")
        raw_mesh = TexturedMesh.from_file(m.raw_model_path)
        raw_mesh.scale(m.scale)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1]) / 2
        if is_transfered:
            translation = [(boxes[i][0] + boxes[i][2]) / 2, boxes[i][4], (boxes[i][1] + boxes[i][3]) / 2, ]
            rotmat = np.zeros((3, 3))
            rotmat[0, 0] = boxes[i][-4]
            rotmat[0, 2] = -boxes[i][-3]
            rotmat[2, 0] = boxes[i][-3]
            rotmat[2, 2] = boxes[i][-4]
            rotmat[1, 1] = 1.

            raw_model.visual.material.image = Image.open(m.texture_image_path)

            raw_model.vertices *= m.scale

            raw_model.vertices -= centroid
            raw_model.vertices = raw_model.vertices.dot(rotmat)
            raw_model.vertices = raw_model.vertices + translation

        obj_out, tex_out = trimesh.exchange.obj.export_obj(
            raw_model,
            return_texture=True
        )

        with open(os.path.join(file_path, names[i]), "w") as f:
            f.write(obj_out.replace("material_0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(file_path, mtl_names[i] + ".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material_0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(file_path, mtl_names[i] + tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def export_floor(file_path, fp):
    raw_floor = trimesh.Trimesh(fp.original_vertices, fp.original_faces, process=True)
    raw_floor.export(f"{file_path}/raw_floor.obj")
    transfer_floor = trimesh.Trimesh(fp.original_vertices - fp.offset, fp.original_faces, process=True)
    transfer_floor.export(f"{file_path}/floor.obj")


def main(argv):
    parser = argparse.ArgumentParser(
        description='Generate scene'
    )

    parser.add_argument(
        "--output_directory",
        default="/data/render_scene/diverse-synth",
        help="path to the output directory"
    )

    parser.add_argument(
        "--room_type",
        default="livingroom",
        choices=[
            "bedroom",
            "livingroom",
            "diningroom",
            "library"
        ],
        help="The type of dataset filtering to be used"
    )

    parser.add_argument(
        "--scene_path",
        default="../dump/",
        help="path to 3D-FUTURE model"
    )

    parser.add_argument(
        "--furniture_path",
        default="../dump/threed_future_model.pkl",
        help="path to 3D-FUTURE model"
    )

    parser.add_argument(
        "--threed_front_dataset_directory",
        default=f'/dataset/3D-FRONT/3D-FRONT/',
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "--threed_future_dataset_directory",
        default=f'/dataset/3D-FRONT//3D-FUTURE-model/',
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "--model_info",
        default=f'/dataset/3D-FRONT/model_info.json',
        help="Path to the 3D-FUTURE model_info.json file"
    )

    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )

    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,-1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0, 4, 0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="1000,1000",
        help="Define the size of the scene and the window"
    )

    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )

    parser.add_argument(
        "--tag",
        default="raw_livingroom",
        help="number of sequences"
    )

    parser.add_argument(
        "--render",
        action='store_true',
        help='whether render scene'
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    config = {
        "room_type_filter": args.room_type,
        "min_n_boxes": -1,
        "max_n_boxes": -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        # "annotation_file": args.annotation_file
    }

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Running code on", device)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if not os.path.exists(f"{args.output_directory}/{args.tag}"):
        os.makedirs(f"{args.output_directory}/{args.tag}")

    plt_render = FloorPlanRenderer()

    object_dataset = FutureDataset.from_pickled_dataset(args.furniture_path)

    dataset = Front.load_dataset(
        dataset_directory=args.threed_front_dataset_directory,
        path_to_model_info=args.model_info,
        path_to_models=args.threed_future_dataset_directory,
        room_type_filter=filter_function(
            config, ["train", "val", "test"], False
        )
    )

    for i in range(len(dataset)):

        scene_idx = i
        current_room = dataset[scene_idx]
        print("{} / {}: using the {} floor plan of scene {}".format(
            i, len(dataset), scene_idx, current_room.scene_id
        ))
        try:
            if current_room.uid in [
                '10c73ad0-214c-4bf7-a4e8-1c154a77b7ed_MasterBedroom-20015',
                '582e09c9-b6ab-41da-bd7e-f8275890dfc1_SecondBedroom-64303',
                '3302e0fc-33e4-47b4-8303-88616dca641b_Bedroom-6144',
                '5106576d-57df-4753-ade6-c87be45f462e_Bedroom-15868',
                '53cf7da8-7c02-419d-bc4a-acf5d4337cac_Bedroom-7405',
                '74dfc90f-42d5-4389-8cbb-5cbe036d99ef_MasterBedroom-37550',
                '8e6f8c6f-a9ac-43a2-946f-18bdc645432c_Bedroom-14389',
                'bdf64124-075e-4aea-a6e4-aa961e3af480_Bedroom-7707',
                'd2f4784e-0fdf-43ea-915b-515f11791cf6_SecondBedroom-23886',
                'f31e12c4-1134-42eb-b28f-2e9e19bc47d6_Bedroom-701',
                'a4a6dbf2-4558-4f48-b567-392202358792_Bedroom-35514',
                '717acc1d-2b59-4dfb-9d9f-6863b548a1ec_SecondBedroom-11545',
                '05ffc5c9-a753-481d-a6b8-bdca64554367_Bedroom-5443'
            ]:
                continue
            fp = FloorPlan(file_path=f"{args.scene_path}/{args.room_type}/{current_room.uid}")
        except Exception as e:
            print(e)
            continue
        room_json = {}
        if not os.path.exists(f"{args.output_directory}/{args.tag}/raw_data"):
            os.mkdir(f"{args.output_directory}/{args.tag}/raw_data")

        data = np.load(f"{args.scene_path}/{args.room_type}/{current_room.uid}/boxes.npz")
        data = data.get('x_abs')
        fp.load_furniture_from_data(data, num_class=23, num_each_class=4, thresholds=0.3)

        data = fp.to_dict(floor_plan_only=False, xyxy=False)

        savemat(f"{args.output_directory}/{args.tag}/raw_data/{i}.mat", data)

        '''render only furniture'''
        if args.render:
            if not os.path.exists(f"{args.output_directory}/{args.tag}/furniture_only"):
                os.mkdir(f"{args.output_directory}/{args.tag}/furniture_only")
            path_to_image = "{}/{}/furniture_only/{}".format(
                args.output_directory,
                args.tag,
                fp.scene_uid,
            )

            if os.path.exists(path_to_image):
                continue

            plt_render.plot_2d_fp_mesh(
                boundary=data['boundary'],
                boxes=data['boxes'],
                types=data['type'],
                colors=data['colors'],
                furniture_only=True,
                with_text=False,
                filename=path_to_image
            )

        '''render floor plan with furniture'''
        if args.render:
            if not os.path.exists(f"{args.output_directory}/{args.tag}/complete_scene"):
                os.mkdir(f"{args.output_directory}/{args.tag}/complete_scene")
            path_to_image = "{}/{}/complete_scene/{}".format(
                args.output_directory,
                args.tag,
                fp.scene_uid,
            )

            if os.path.exists(path_to_image):
                continue

            plt_render.plot_2d_fp_mesh(
                boundary=data['boundary'],
                boxes=data['boxes'],
                types=data['type'],
                colors=data['colors'],
                furniture_only=False,
                with_text=False,
                filename=path_to_image
            )
        model_list, furniture_list, room_params, boxes = get_furniture_by_params(
            data['boxes'], data["lamps"], object_dataset, fp.scene_type, fp.scene_id)

        mesh = boundary2mesh(fp)
        room_json['furniture'] = furniture_list
        room_json['mesh'] = [mesh]
        room_json['scene'] = [room_params]
        file_path = f"{args.output_directory}/{args.tag}/renderable_scene/{i}"
        if not os.path.exists(f"{args.output_directory}/{args.tag}/renderable_scene"):
            os.mkdir(f"{args.output_directory}/{args.tag}/renderable_scene")
        if not os.path.exists(f"{args.output_directory}/{args.tag}/renderable_scene/{i}"):
            os.mkdir(f"{args.output_directory}/{args.tag}/renderable_scene/{i}")
        json_str = json.dumps(room_json, indent=4)
        with open(f'{file_path}/house.json', 'w') as json_file:
            json_file.write(json_str)

        if not os.path.exists(f"{args.output_directory}/{args.tag}/renderable_scene/{i}/raw"):
            os.mkdir(f"{args.output_directory}/{args.tag}/renderable_scene/{i}/raw")
        if not os.path.exists(f"{args.output_directory}/{args.tag}/renderable_scene/{i}/transform"):
            os.mkdir(f"{args.output_directory}/{args.tag}/renderable_scene/{i}/transform")
        try:
            export_scene(f"{file_path}/transform", model_list, boxes, is_transfered=True)
            export_scene(f"{file_path}/raw", model_list, boxes, is_transfered=False)
            export_floor(file_path, fp)
        except Exception:
            print("{} / {}: retry the floor plan of scene {}".format(
                i, len(dataset), fp.scene_id
            ))


if __name__ == '__main__':
    main(sys.argv[1:])
