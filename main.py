from server import *
from data_owner import *
from client import *

if __name__ == '__main__':
    # 数据持有者初始化并外包数据
    data_owner()
    # 服务器查询
    server(k, start, query_vector)
    # 客户端验证
    print(client_verified())
