import io
import struct

import numpy as np


def read_fvecs(file_path, dim_given=None):
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            # 读取4字节整数，表示向量的维度
            dim_byte = f.read(4)
            if not dim_byte:
                break
            dim = struct.unpack('i', dim_byte)[0]

            # 读取浮点数值并存储为numpy数组
            vector = np.zeros(dim, dtype=np.float32)
            for i in range(dim):
                float_byte = f.read(4)
                vector[i] = struct.unpack('f', float_byte)[0]
            vectors.append(vector)
        if dim_given:
            return np.array(vectors)[:, :dim_given]

    return np.array(vectors)


def read_fvecs_fast(file_path, dim_given=None):
    with open(file_path, 'rb') as f:
        # 读取4字节整数，表示向量的维度
        dim_byte = f.read(4)
        dim = struct.unpack('i', dim_byte)[0]

        # 读取浮点数值并存储为numpy数组
        vector = np.fromfile(f, dtype=np.float32, count=dim)
        vectors = [vector]

        while True:
            dim_byte = f.read(4)
            if not dim_byte:
                break
            dim = struct.unpack('i', dim_byte)[0]
            vector = np.fromfile(f, dtype=np.float32, count=dim)
            vectors.append(vector)

    if dim_given:
        return np.array(vectors)[:, :dim_given]
    else:
        return np.array(vectors)


def read_glove(file_path):
    vectors = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.rstrip('\n').split()
            if len(tokens) == 101:
                vector = np.array(tokens[1:], dtype=np.float32)
                vectors.append(vector)

    vectors = np.array(vectors)[:1000000, :]
    print(vectors.shape)
    return vectors
