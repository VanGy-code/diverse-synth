import argparse
import logging
import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm.contrib import tzip

from synthesis.datasets.FloorPlan import FloorPlan
from synthesis.datasets.FRONT import Front
from synthesis.datasets.Common import filter_function
from synthesis.datasets.Utils import parse_threed_front_scenes
from synthesis.datasets.FrontDataset import FrontFactory

from visulization import render, scene_from_args, FloorPlanRenderer
from scripts import ensure_parent_directory_exists

def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )

    parser.add_argument(
        "--output_directory",
        default="../dump/",
        help="Path to output directory"
    )
    parser.add_argument(
        "--threed_front_dataset_directory",
        default=f'/dataset/3D-FRONT/3D-FRONT/',
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "--threed_future_dataset_directory",
        default=f'/dataset/3D-FRONT/3D-FUTURE-model/',
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
    # parser.add_argument(
    #     "--annotation_file",
    #     default="../config/bedroom_threed_front_splits.csv",
    #     help="Path to the train/test splits file"
    # )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )

    parser.add_argument(
        "--render",
        action='store_true',
        help='whether render scene'
    )

    parser.add_argument(
        "--dataset_filtering",
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
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )

    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="1000,1000",
        help="Define the size of the scene and the window"
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
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )

    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = {
        "room_type_filter": args.dataset_filtering,
        "min_n_boxes": -1,
        "max_n_boxes": -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        # "annotation_file": args.annotation_file
    }

    plt_render = FloorPlanRenderer()

    dataset = Front.load_dataset(
        dataset_directory=args.threed_front_dataset_directory,
        path_to_model_info=args.model_info,
        path_to_models=args.threed_future_dataset_directory,
        room_type_filter=filter_function(config, ["train", "val"], args.without_lamps)
    )

    print("Loading dataset with {} rooms".format(len(dataset)))

    trans_bounds = dataset.bounds["translations"]
    size_bounds = dataset.bounds["sizes"]
    angle_bounds = dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": trans_bounds[0].tolist() + trans_bounds[1].tolist(),
        "bounds_sizes": size_bounds[0].tolist() + size_bounds[1].tolist(),
        "bounds_angles": angle_bounds[0].tolist() + angle_bounds[1].tolist(),
        "class_labels": dataset.class_labels,
        "class_order": dataset.class_order,
        "furniture_limit": dataset.furniture_limit,
    }

    if args.render:
        scene = scene_from_args(args)

    room_type_directory = os.path.join(args.output_directory, config["room_type_filter"])
    if os.path.exists(room_type_directory) is False:
        os.mkdir(room_type_directory)

    path_to_json = os.path.join(args.output_directory, f"{config['room_type_filter']}/dataset_stats.txt")
    with open(path_to_json, "w") as f:
        json.dump(dataset_stats, f)
        
    print(
        "Saving training statistics for dataset with bounds: {} to {}".format(
            dataset.bounds, path_to_json
        )
    )
    
    dataset = Front.load_dataset(
        dataset_directory=args.threed_front_dataset_directory,
        path_to_model_info=args.model_info,
        path_to_models=args.threed_future_dataset_directory,
        room_type_filter=filter_function(
            config, ["train", "val", "test"], args.without_lamps
        )
    )

    print("Loading dataset with {} rooms".format(len(dataset)))

    encoded_dataset = FrontFactory.encode_dataset(
        name="basic", dataset=dataset)

    for room_id, room in tzip(encoded_dataset, dataset):
        # Create a separate folder for each room

        room_directory = f"{room_type_directory}/{room.uid}"
        # Check if room_directory exists and if it doesn't create it
        if os.path.exists(room_directory):
            continue
        
        ensure_parent_directory_exists(room_directory)

        uids = [bi.model_uid for bi in room.bboxes]
        jids = [bi.model_jid for bi in room.bboxes]

        floor_plan_vertices, floor_plan_faces = room.floor_plan

        # Render and save the room mask as an image
        if args.render:
            floor_plan = room.floor_plan_renderable(with_door=False)
            room_mask = render(
                scene,
                floor_plan,
                color=(1.0, 1.0, 1.0),
                mode="flat",
                frame_path=f"{room_directory}/room_mask.png"
            )[:, :, 0:1]

        np.savez_compressed(
            f"{room_directory}/boxes",
            uids=uids,
            jids=jids,
            scene_id=room.scene_id,
            scene_uid=room.uid,
            scene_type=room.scene_type,
            json_path=room.json_path,
            floor_plan_vertices=floor_plan_vertices,
            floor_plan_faces=floor_plan_faces,
            floor_plan_centroid=room.floor_plan_centroid,
            x_abs=room_id["x_abs"],
        )

        with open(f"{room_directory}/rel.pkl", 'wb') as f:
            pickle.dump(room_id["x_rel"], f)


if __name__ == '__main__':
    debug = False
    main(sys.argv[1:])
