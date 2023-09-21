import pickle

import numpy as np
from collections import OrderedDict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

from modules.HCNNG.guide_search import *
from modules.outsource.VO import *
from modules.outsource.verified_search import verified_search

# 客户端提供查询条件
k = 5
start = 5
query_vector = np.array([0, 16, 35, 5, 32, 31, 14, 10, 11, 78, 55, 10, 45, 83,
                         11, 6, 14, 57, 102, 75, 20, 8, 3, 5, 67, 17, 19, 26,
                         5, 0, 1, 22, 60, 26, 7, 1, 18, 22, 84, 53, 85, 119,
                         119, 4, 24, 18, 7, 7, 1, 81, 106, 102, 72, 30, 6, 0,
                         9, 1, 9, 119, 72, 1, 4, 33, 119, 29, 6, 1, 0, 1,
                         14, 52, 119, 30, 3, 0, 0, 55, 92, 10, 2, 5, 4, 9,
                         22, 89, 96, 14, 1, 0, 1, 82, 59, 16, 25, 5, 26, 158,
                         11, 4, 0, 0, 1, 26, 47, 23, 4, 0, 0, 4, 38, 83,
                         30, 14, 9, 4, 9, 17, 23, 41, 0, 0, 2, 8, 19, 25,
                         23, 1],
                        dtype=np.float32)


def client_verified():
    with open("./communication_file/vos.pkl", "rb") as f:
        vos = pickle.load(f)

    with open("./communication_file/knn.pkl", "rb") as f:
        knn = pickle.load(f)

    with open("./communication_file/signature.pkl", "rb") as f:
        signature = pickle.load(f)

    with open("./communication_file/root_hash_DO.pkl", "rb") as f:
        root_hash_DO = pickle.load(f)

    with open("./communication_file/public_key.pem", "rb") as f:
        pem_public = f.read()

    # 反序列化公钥
    public_key = serialization.load_pem_public_key(
        pem_public,
        backend=default_backend()
    )

    if len(knn) != k:
        print("k")
        return False

    try:
        public_key.verify(
            signature,
            root_hash_DO,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    except InvalidSignature:
        print("sign")
        return False

    gts = dict()
    vectors = dict()
    for vo in vos:
        if not vo.is_hash:
            idx_gt_vector = json_tricks.loads(vo.data)
            index = idx_gt_vector[0]
            gt = idx_gt_vector[1]
            vector = idx_gt_vector[2]
            gts[index] = gt
            vectors[index] = vector

    tries = build_dict_tries(gts)


    root_hash_Client = bytes.fromhex(vo_compute(vos).data)

    if root_hash_Client != root_hash_DO:
        print("vo")
        return False

    _, knn_client = verified_search(vectors, tries, gts, k, start, query_vector)

    if knn_client != knn:
        return False

    return True


# print(client_verified())
