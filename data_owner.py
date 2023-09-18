import math
import pickle

import numpy as np

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

from modules.HCNNG.guide_search import get_gts
from modules.HCNNG.hcnng import *
from modules.HCNNG.load_dataset import read_fvecs
from modules.outsource.MHT import *


def data_owner():
    vectors = read_fvecs("./resource/siftsmall/siftsmall_base.fvecs")
    num_vertices = vectors.shape[0]
    indexes = range(num_vertices)
    hcnng = createHCNNG(vectors, indexes, int(math.sqrt(num_vertices)), 20)
    gts = get_gts(vectors, hcnng)
    hash_list = gts_to_hash(gts, vectors)
    root_hash_DO = bytes.fromhex(get_merkle_root(hash_list))

    # 生成RSA密钥对
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # 序列化公钥和私钥
    pem_private = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    pem_public = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # 使用私钥进行数字签名
    signature = private_key.sign(
        root_hash_DO,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    with open("./communication_file/vectors.pkl", "wb") as f:
        pickle.dump(vectors, f)

    with open("./communication_file/hcnng.pkl", "wb") as f:
        pickle.dump(hcnng, f)

    with open("./communication_file/gts.pkl", "wb") as f:
        pickle.dump(gts, f)

    with open("./communication_file/root_hash_DO.pkl", "wb") as f:
        pickle.dump(root_hash_DO, f)

    with open("./communication_file/signature.pkl", "wb") as f:
        pickle.dump(signature, f)

    with open("./communication_file/public_key.pem", "wb") as f:
        f.write(pem_public)
