import cProfile
import json
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from modules.HCNNG.load_dataset import *
from server import *
from data_owner import *
from client import *
from modules.HCNNG.hcnng import createHCNNG
import os


def get_file_size_in_mb(file_path):
    try:
        # 获取文件大小（字节）
        file_size_bytes = os.path.getsize(file_path)

        # 将文件大小转换为MB
        file_size_mb = file_size_bytes / (1024 * 1024)

        return file_size_mb
    except FileNotFoundError:
        return "文件未找到"


def exp1():
    # 数据持有者初始化并外包数据
    # data_owner(dataset="gist")
    # 服务器查询
    index = list(range(10, 110, 10))
    df = pd.DataFrame(index=index, columns=['vo_size'])

    with open("./communication_file/vectors.pkl", "rb") as f:
        vectors = pickle.load(f)

    with open("./communication_file/gts.pkl", "rb") as f:
        gts = pickle.load(f)

    print("read down")

    tries = build_tries(gts)
    print("build tries")

    with open("./communication_file/tires.pkl", "wb") as f:
        pickle.dump(tries, f)

    for k in index:
        print(f"寻找{k}个近邻")

        visited, knn = verified_search(vectors, tries, gts, k, start, query_vector)
        print("find knn")

        vos = vo_construction(gts, visited, vectors)
        print("vo construct")

        with open("./communication_file/vos.pkl", "wb") as f:
            pickle.dump(vos, f)

        file_path = rf"D:\project\HCNNG\communication_file\vos.pkl"
        file_size_mb = get_file_size_in_mb(file_path)
        print(f"文件大小：{file_size_mb} MB")
        df.loc[k] = file_size_mb

    df.index.name = 'k'
    df.to_excel("k-vosize.xlsx", index=True, sheet_name='Sheet1')
    # 客户端验证
    # print(client_verified())


def exp2():
    vectors = np.load("./resource/glove/glove.npy")
    num_vertices = vectors.shape[0]
    print(vectors.shape)
    indexes = range(num_vertices)

    hcnng = createHCNNG_parallel(vectors, indexes, 2000, 20)

    neighborss = get_all_neighbors(hcnng, vectors.shape[0])

    gts = get_gts(vectors, hcnng)
    tries = build_dict_tries(gts)

    ks = range(10, 110, 10)
    size_vos_gts = []
    size_vos_nos = []
    size_vos_trees = []
    for k in ks:
        print(k)
        visited1, knn1 = verified_search(vectors, tries, gts, k, start, query_vector)
        vos_gt, _ = vo_construction_simple(gts, visited1, vectors)

        visited2, knn2 = verified_search_without_guide(vectors, neighborss, k, start, query_vector)
        vos_no, _ = vo_construction_simple(gts, visited2, vectors)

        vos_tree = vo_construction_with_tries_simple(tries, gts, visited1, vectors)

        with open("./exp_data/vos_gt.pkl", "wb") as f:
            pickle.dump(vos_gt, f)

        with open("./exp_data/vos_no.pkl", "wb") as f:
            pickle.dump(vos_no, f)

        with open("./exp_data/vos_tree.pkl", "wb") as f:
            pickle.dump(vos_tree, f)

        size_vos_gt = os.path.getsize("./exp_data/vos_gt.pkl") / (1024)
        size_vos_no = os.path.getsize("./exp_data/vos_no.pkl") / (1024)
        size_vos_tree = os.path.getsize("./exp_data/vos_tree.pkl") / (1024)

        size_vos_gts.append(size_vos_gt)
        size_vos_nos.append(size_vos_no)
        size_vos_trees.append(size_vos_tree)

    df = pd.DataFrame(
        {
            "index": ks,
            "size_vos_gt": size_vos_gts,
            "size_vos_no": size_vos_nos,
            "size_vos_tree": size_vos_trees
        }
    )
    df.set_index('index', inplace=True)

    df.to_excel("glove-k-vosize.xlsx")


def exp3(dataset):
    num_con_comp_gts = []
    num_con_comp_nos = []

    num_decon_comp_gts = []
    num_decon_comp_nos = []

    k = 20
    vectors_meta = np.load(rf"./resource/{dataset}/{dataset}.npy")
    print(f"vectors_meta:{vectors_meta.shape}")
    s = int(vectors_meta.shape[0] / 10)
    num_range = list(range(s, vectors_meta.shape[0], s))
    for num in num_range:
        vectors = vectors_meta[:num, :]

        num_vertices = vectors.shape[0]
        indexes = range(num_vertices)
        hcnng = createHCNNG_parallel(vectors, indexes, 2000, 20)
        neighborss = get_all_neighbors(hcnng, vectors.shape[0])
        gts = get_gts(vectors, hcnng)
        tries = build_dict_tries(gts)

        visited1, knn1 = verified_search(vectors, tries, gts, k, start, query_vector)
        vos_gt, num_con_comp_gt = vo_construction_simple(gts, visited1, vectors)

        visited2, knn2 = verified_search_without_guide(vectors, neighborss, k, start, query_vector)
        vos_no, num_con_comp_no = vo_construction_simple(gts, visited2, vectors)

        _, num_decon_comp_gt = vo_compute_simple(vos_gt)
        _, num_decon_comp_no = vo_compute_simple(vos_no)

        num_con_comp_gts.append(num_con_comp_gt)
        num_con_comp_nos.append(num_con_comp_no)
        num_decon_comp_gts.append(num_decon_comp_gt)
        num_decon_comp_nos.append(num_decon_comp_no)

    df = pd.DataFrame(
        {
            "index": num_range,
            "num_con_comp_gts": num_con_comp_gts,
            "num_con_comp_nos": num_con_comp_nos,
            "num_decon_comp_gts": num_decon_comp_gts,
            "num_decon_comp_nos": num_decon_comp_nos
        }
    )
    df.set_index('index', inplace=True)
    df.to_excel(rf"{dataset}-datasize-computeNum.xlsx")


def exp4(dataset):
    size_vos_gts = []
    size_vos_nos = []

    vectors = np.load(rf"./resource/{dataset}/{dataset}.npy")
    print(f"vectors_meta:{vectors.shape}")
    num_vertices = vectors.shape[0]
    indexes = range(num_vertices)
    hcnng = createHCNNG_parallel(vectors, indexes, 2000, 20)
    neighborss = get_all_neighbors(hcnng, vectors.shape[0])
    gts = get_gts(vectors, hcnng)
    tries = build_dict_tries(gts)

    visited_muti1 = set()
    visited_muti2 = set()

    query_num = range(0, 100)

    for start in query_num:
        visited1, knn1 = verified_search(vectors, tries, gts, k, start, query_vector)
        visited_muti1.update(visited1)

        visited2, knn2 = verified_search_without_guide(vectors, neighborss, k, start, query_vector)
        visited_muti2.update(visited2)

        if start != 0 and start % 10 == 0:
            vos_gt, _ = vo_construction_simple(gts, visited1, vectors)
            with open("./exp_data/vos_gt.pkl", "wb") as f:
                pickle.dump(vos_gt, f)

            vos_no, _ = vo_construction_simple(gts, visited2, vectors)
            with open("./exp_data/vos_no.pkl", "wb") as f:
                pickle.dump(vos_no, f)

            size_vos_gt = os.path.getsize("./exp_data/vos_gt.pkl") / (1024)
            size_vos_no = os.path.getsize("./exp_data/vos_no.pkl") / (1024)
            size_vos_gts.append(size_vos_gt)
            size_vos_nos.append(size_vos_no)

    df = pd.DataFrame(
        {
            "qn": query_num,
            "size_vos_gt": size_vos_gts,
            "size_vos_no": size_vos_nos,
        }
    )
    df.set_index('qn', inplace=True)

    df.to_excel(f"{dataset}-qn-vosize.xlsx")


def exp5(dataset):
    time_tree=[]
    time_no_guide=[]

    vectors = np.load(rf"./resource/{dataset}/{dataset}.npy")
    print(f"vectors_meta:{vectors.shape}")
    num_vertices = vectors.shape[0]
    indexes = range(num_vertices)
    hcnng = createHCNNG_parallel(vectors, indexes, 2000, 20)
    neighborss = get_all_neighbors(hcnng, vectors.shape[0])
    gts = get_gts(vectors, hcnng)
    tries = build_dict_tries(gts)

    ks = range(10, 110, 10)

    for k in ks:
        print(k)
        visited1, knn1, ndc1 = verified_search(vectors, tries, gts, k, start, query_vector)
        print(rf"ndc1:{ndc1}")
        time_tree.append(ndc1)
        visited2, knn2, ndc2 = verified_search_without_guide(vectors, neighborss, k, start, query_vector)
        print(rf"ndc1:{ndc2}")
        time_no_guide.append(ndc2)


    df = pd.DataFrame(
        {
            "index": ks,
            "time_guide": time_tree,
            "time_no_guide": time_no_guide,
        }
    )
    df.set_index('index', inplace=True)

    df.to_excel(rf"{dataset}-k-speedup.xlsx")


if __name__ == '__main__':
    exp5("sift")
    exp5("gist")
    exp5("glove")