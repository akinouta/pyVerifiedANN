import pandas as pd

from server import *
from data_owner import *
from client import *
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


if __name__ == '__main__':
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


