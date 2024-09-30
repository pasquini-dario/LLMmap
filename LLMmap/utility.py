import re
import pickle
import hashlib
import os, glob, random
import re
import tensorflow as tf

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        ...

def _hash(input_string):
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    integer_hash = int(sha256_hash, 16)
    return integer_hash

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(path, data):
    with open(path, 'wb') as f:
        data = pickle.dump(data, f)

def sample_from_multi_universe(universe):
    sample = {}
    for k, u in universe.items():
        sample[k] = random.sample(u, 1)[0]
    return sample


def set_gpus(gpus, with_tf_gpu_memory_growth=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    if with_tf_gpu_memory_growth:
        tf_gpu_memory_growth()

def tf_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
