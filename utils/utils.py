import numpy as np
import os

def normalize(x, b=None, u=None):
    if b != None:
        x -= b
    else:
        x -= np.min(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if u != None:
            x /= u
        else:
            x /= np.max(x)
    x = np.nan_to_num(x)
    # x *= 2
    # x -= 1
    return x


def scale(x, min=0, max=1):
    # scale x to [min, max]
    x = np.nan_to_num(x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    return x


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
