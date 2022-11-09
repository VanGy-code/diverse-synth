import argparse
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthesis.datasets import THREED_FRONT_FURNITURE
from synthesis.datasets.Common import filter_function
from synthesis.datasets.FRONT import Front
from synthesis.datasets.FutureDataset import FutureDataset


def main(argv):
    parser = argparse.ArgumentParser(
        description="Pickle the 3D Future dataset"
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
        default=f'/dataset/3D-FRONT/3D-FUTURE-model',
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "--model_info",
        default=f'/dataset/3D-FRONT/model_info.json',
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )
    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )

    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )

    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = {
        "room_type_filter": "no_filtering",
        "min_n_boxes": -1,
        "max_n_boxes": -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
    }

    scenes_dataset = Front.load_dataset(
        dataset_directory=args.threed_front_dataset_directory,
        path_to_model_info=args.model_info,
        path_to_models=args.threed_future_dataset_directory,
        room_type_filter=filter_function(config, ["train", "val", "test"], args.without_lamps)
    )
    print("Loading dataset with {} rooms".format(len(scenes_dataset)))


    objects = {}
    for scene in scenes_dataset:
        for obj in scene.bboxes:
            if obj.label is not None:
                objects[obj.model_jid] = obj
    objects = [vi for vi in objects.values()]

    objects_dataset = FutureDataset(objects)
    output_path = "{}/threed_future_model.pkl".format(
        args.output_directory,
    )
    with open(output_path, "wb") as f:
        pickle.dump(objects_dataset, f)
        
    print(objects_dataset.labels)
    print(len(objects_dataset.labels))


if __name__ == "__main__":
    main(sys.argv[1:])



