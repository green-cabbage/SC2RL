
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(data : dict, file : str):

    with open(file, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

def load_json(file : str) -> dict:

    with open(file, "r") as f:
        data = json.load(f)
        return data
