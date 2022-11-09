import numpy as np
import matlab
import matlab.engine as engine
import os

eng = engine.start_matlab()
eng.addpath(os.path.join(os.path.dirname(__file__), 'matlab'), nargout=0)

def align_fp(boundary, boxes, types, edges, cmap, threshold):
    boundary = np.array(boundary, dtype=float).tolist()
    boxes = np.array(boxes, dtype=float).tolist()
    types = np.array(types, dtype=int).tolist()
    edges = np.array(edges, dtype=int).tolist()
    cmap = np.array(cmap, dtype=float).tolist()

    boxes_aligned, types = eng.align_fp(
        matlab.double(boundary),
        matlab.double(boxes),
        matlab.double(types),
        matlab.double(edges),
        threshold,
        matlab.double(cmap),
        False,
        nargout=2
    )

    boxes_aligned = np.array(boxes_aligned, dtype=float)
    types = np.array(types, dtype=int).reshape(-1)

    return boxes_aligned, types
