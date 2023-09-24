import json

from .data_structure import *
from .MHT import *
from ..HCNNG.util import tree_to_dict


def vo_construction(gts, visited, vectors):
    vos = []
    num_vertices = len(gts)

    # leaf node
    for index, gt in enumerate(gts):
        if index in visited:
            vos.append(
                VO(
                    json_tricks.dumps((index, gts[index], vectors[index])),
                    False,
                    0,
                    index
                )
            )
        else:
            vos.append(
                VO(
                    gt_to_hash(index, gt, vectors[index]),
                    True,
                    0,
                    index
                )
            )

    # merge
    vo_high = 1
    while vo_high < num_vertices:
        new_vos = []
        num_vos = len(vos)
        index = 0

        while index < num_vos:
            # 可以合并两个叶子节点
            if (index + 1 < num_vos and vos[index].site % 2 == 0 and
                    vos[index].is_hash and vos[index + 1].is_hash and
                    vos[index].level == vos[index + 1].level):
                new_vos.append(
                    VO(
                        hash_data(vos[index].data + vos[index + 1].data),
                        True,
                        vos[index].level + 1,
                        vos[index].site // 2  # 向下取整
                    )
                )
                index += 1
            # 不可以合并
            else:
                new_vos.append(vos[index])

            index += 1

        vos = new_vos
        vo_high *= 2

    return vos


def vo_compute(vos):
    for vo in vos:
        if not vo.is_hash:
            vo.data = hash_data(vo.data)
            vo.is_hash = True

    while len(vos) != 1:

        new_vos = []
        num_vos = len(vos)

        index = 0

        while index < num_vos:
            # 向上合并
            if index + 1 < num_vos and vos[index].site % 2 == 0 and vos[index].level == vos[index + 1].level:
                new_vos.append(
                    VO(
                        hash_data(vos[index].data + vos[index + 1].data),
                        True,
                        vos[index].level + 1,
                        vos[index].site // 2
                    )
                )
                index += 1
            else:
                new_vos.append(vos[index])

            index += 1

        last = len(new_vos) - 1
        if last > 0 and new_vos[last].level < new_vos[last - 1].level:
            difference = new_vos[last - 1].level - new_vos[last].level
            new_vos[last].level = new_vos[last - 1].level
            # new_vos[last].site /= 2 ** difference
            for i in range(difference):
                new_vos[last].data = hash_data(new_vos[last].data)
                new_vos[last].site //= 2

        vos = new_vos

    return vos[0]


def vo_construction_with_tries(tries, gts, visited, vectors):
    vos = []
    num_vertices = len(gts)

    # leaf node
    for index, gt in enumerate(gts):
        if index in visited:
            vos.append(
                VO(
                    json_tricks.dumps((index, gts[index], vectors[index], json.dumps(tree_to_dict(tries[index])))),
                    False,
                    0,
                    index
                )
            )
        else:
            vos.append(
                VO(
                    gt_to_hash(index, gt, vectors[index]),
                    True,
                    0,
                    index
                )
            )

    # merge
    vo_high = 1
    while vo_high < num_vertices:
        new_vos = []
        num_vos = len(vos)
        index = 0

        while index < num_vos:
            # 可以合并两个叶子节点
            if (index + 1 < num_vos and vos[index].site % 2 == 0 and
                    vos[index].is_hash and vos[index + 1].is_hash and
                    vos[index].level == vos[index + 1].level):
                new_vos.append(
                    VO(
                        hash_data(vos[index].data + vos[index + 1].data),
                        True,
                        vos[index].level + 1,
                        vos[index].site // 2  # 向下取整
                    )
                )
                index += 1
            # 不可以合并
            else:
                new_vos.append(vos[index])

            index += 1

        vos = new_vos
        vo_high *= 2

    return vos
