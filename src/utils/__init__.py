from os.path import isfile
from typing import List, Dict


def key_exists(data_dict: Dict, keys: List):
    for key in keys:
        if key in data_dict:
            data_dict = data_dict[key]
        else:
            return False, None
    return True, data_dict


def get_unique_fname(base_name, extension, max_postfix=None):
    postfix = 0
    while True or (max_postfix is not None and postfix < max_postfix):
        fname = f'{base_name}_{postfix}.{extension}'
        if not isfile(fname):
            return fname
        postfix += 1
    return None
