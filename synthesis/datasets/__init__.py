import csv
import numpy as np

THREED_FRONT_FURNITURE = {
    "desk": "desk", # 0
    "nightstand": "nightstand", # 1
    'double bed': "bed", # 2
    'bed frame': "bed", # 2
    'bunk bed': "bed", # 2
    "king-size bed": "bed", # 2
    "single bed": "bed", # 2
    "kids bed": "bed", # 2
    "bunk bed": "bed", # 2
    "couch bed": "bed", # 2
    "wardrobe": "wardrobe", # 3
    "bookcase/jewelry armoire": "shelf", # 4
    "shelf": "shelf", # 4
    "tv stand": "tv_stand", # 5
    "lounge chair/cafe chair/office chair": "chair", # 6
    'lounge chair/book-chair/computer chair': "chair", # 6
    "armchair": "chair", # 6
    "hanging chair": "chair", # 6
    'folding chair': 'chair', # 6
    "classic chinese chair": "classic_chair", # 7
    "dining chair": "dining_chair", # 8
    "dressing chair": "dressing_chair", # 9
    "dressing table": "dressing_table", # 10
    "dining table": "dining_table", # 11
    "coffee table": "coffee_table/tea_table", # 12
    "tea table": "coffee_table/tea_table", # 12
    "corner/side table": "end_table", # 13
    "round end table": "end_table", # 13
    "barstool": "bar", # 14
    'bar': 'bar', # 14
    "cabinet": "cabinet", # 15
    "drawer chest/corner cabinet": "cabinet", # 15
    "sideboard/side cabinet/console table": "cabinet",# 15
    "sideboard/side cabinet/console": "cabinet", # 15
    "children cabinet": "children_cabinet", # 16
    'wine cooler': "wine_cabinet", # 17
    "wine cabinet": "wine_cabinet", # 17
    "shoe cabinet": "shoe_cabinet", # 18
    "footstool/sofastool/bed end stool/stool": "stool", # 19
    "loveseat sofa": "sofa", # 20
    'two-seat sofa': 'sofa', # 20
    "three-seat/multi-seat sofa": "sofa", # 20
    "three-seat/multi-person sofa": "sofa", # 20
    "l-shaped sofa": "sofa", # 20
    "u-shaped sofa": "sofa", # 20
    "lazy sofa": "sofa", # 20
    "chaise longue sofa": "sofa", # 20
    'couch bed': 'sofa', # 20
    'floor lamp': 'floor_lamp', # 21
    'wall lamp': 'lamp', #21
    "ceiling lamp": "lamp", # 22
    "pendant lamp": "lamp", # 22
}

class SplitsBuilder(object):
    def __init__(self, train_test_splits_file):
        self._train_test_splits_file = train_test_splits_file
        self._splits = {}

    def train_split(self):
        return self._splits["train"]

    def test_split(self):
        return self._splits["test"]

    def val_split(self):
        return self._splits["val"]


class CSVSplitsBuilder(SplitsBuilder):
    def __init__(self, train_test_splits_file):
        super().__init__(train_test_splits_file)

    def _parse_train_test_splits_file(self):
        with open(self._train_test_splits_file, "r") as f:
            data = [row for row in csv.reader(f)]
        return np.array(data)

    def get_splits(self, keep_splits=None):
        if keep_splits is None:
            keep_splits = ["train, val"]
        if not isinstance(keep_splits, list):
            keep_splits = [keep_splits]
        # Return only the split
        s = []
        for ks in keep_splits:
            s.extend(self._parse_split_file()[ks])
        return s

    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            for s in ["train", "test", "val"]:
                self._splits[s] = [r[0] for r in data if r[1] == s]
        return self._splits

