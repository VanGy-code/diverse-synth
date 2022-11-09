import numpy as np

from synthesis.datasets import THREED_FRONT_FURNITURE


def get_furniture_by_params(bbox_params, lamps, object_dataset, scene_type, instance_id):
    """
        furniture_item = {
            "uid": None, "jid": None, "aid": [], "size": [],
            "sourceCategoryId": None, "title": None, "category": None,
            "bbox": [], "valid": True,
        }

        room_params = {
            "ref": -1, "pos": None, "rot": None, "scale": None, "room": []
        }
    """
    model_list = []
    furniture_list = []
    room_params = {
        "type": scene_type.tolist()[0],
        "instanceid": instance_id,
        "pos": [
            0,
            0,
            0
        ],
        "rot": [
            0,
            0,
            0,
            1
        ],
        "scale": [
            1,
            1,
            1
        ]
    }

    children = []

    cat2id = dict()
    index = 0
    for d in [THREED_FRONT_FURNITURE]:
        for k, v in d.items():
            if v not in cat2id.keys():
                cat2id[v] = index
                index += 1
    id2cat = {val: key for key, val in cat2id.items()}

    boxes = bbox_params.tolist()
    lamps = lamps.tolist()
    boxes.extend(lamps)

    invalid_model_list = [
        '7e101ef3-7722-4af8-90d5-7c562834fabd'
    ]

    for j in range(len(boxes)):
        query_size = np.array([boxes[j][6], boxes[j][5], boxes[j][7]])

        query_label_index = boxes[j][-1]
        query_label = id2cat.get(query_label_index)
        try:
            furniture = object_dataset.get_closest_furniture_to_box(
                query_label, query_size, invalid_model_list
            )
        except Exception:
            raise

        model_list.append(furniture)

        furniture_param = {
            "uid": furniture.model_uid,
            "jid": "object_{:03d}.obj".format(j),
            "aid": [],
            "size": furniture.size.tolist(),
            "title": furniture.instance_id,
            "category": furniture.label,
            "bbox": furniture.gen_box_from_params().tolist(),
            "valid": True,
        }

        scene_object_param = {
            "ref": furniture.model_uid,
            "pos": [(boxes[j][0] + boxes[j][2]) / 2, boxes[j][4], (boxes[j][1] + boxes[j][3]) / 2, ],
            "scale": furniture.scale,
            "rot": boxes[j][-4:-1],
            "theta": np.arccos(boxes[j][-4]),
            "instanceid": furniture.instance_id
        }

        furniture_list.append(furniture_param)
        children.append(scene_object_param)

    room_params["children"] = children

    return model_list, furniture_list, room_params, boxes
