import hashlib
import json_tricks


def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()


def get_merkle_root(hash_list):
    while len(hash_list) > 1:
        temp_list = []
        for i in range(0, len(hash_list), 2):
            combined_data = hash_list[i]
            if i + 1 < len(hash_list):
                combined_data += hash_list[i + 1]
            temp_list.append(hash_data(combined_data))
        hash_list = temp_list
    return hash_list[0] if hash_list else None


def vectors_to_str(vectors):
    return [",".join(vector.astype(str)) for vector in vectors]


def vectors_to_hash(str_vectors):
    return [hash_data(str_vector) for str_vector in str_vectors]


def gt_to_hash(index, gt, vector):
    return hash_data(json_tricks.dumps((index, gt, vector)))


def gts_to_hash(gts, vectors):
    return [gt_to_hash(index, gt, vectors[index]) for index, gt in enumerate(gts)]
