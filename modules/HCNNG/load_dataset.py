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
            print(dim_given)
            return np.array(vectors)[:, :dim_given]

    return np.array(vectors)

# # 使用函数读取.fvecs文件
# file_path = r'../../resource/siftsmall/siftsmall_base.fvecs'
# vectors = read_fvecs(file_path)
# print(vectors)
# print(vectors.shape)
