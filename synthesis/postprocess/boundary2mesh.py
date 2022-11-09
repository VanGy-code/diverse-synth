def boundary2mesh(fp):
    return {
        "aid": [],
        "jid": "temp",
        "uid": "temp",
        "offset": fp.offset.tolist(),
        "xyz": fp.original_vertices.tolist(),
        "uv": [],
        "faces": fp.original_faces.tolist(),
        "material": None,
        "type": "Floor"
    }
