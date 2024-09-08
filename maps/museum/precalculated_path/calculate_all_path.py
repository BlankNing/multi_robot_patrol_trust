import os
import numpy as np
import itertools
from utils.load_map import *
from utils.astar_shortest_path import calculate_shortest_path
from tqdm import tqdm  # 导入tqdm库来显示进度条
from multiprocessing import Pool, cpu_count  # 导入多进程库

map_name = 'museum'

def read_pgm_image(file_path):
    with open(file_path, 'rb') as f:
        img = Image.open(f)
        img_array = np.array(img)
    return img_array

node_pos_matrix = np.load(f'./{map_name}_node_positions.npy')
map_adj_matrix = np.load(f'./{map_name}_adj_matrix.npy')
pgm_map_matrix = read_pgm_image(f'./{map_name}.pgm')
nodes_num = len(node_pos_matrix)

def compute_path_for_pair(pair):
    """
    计算给定兴趣点对之间的路径。

    参数:
    - pair: 元组 (start_node, end_node)

    返回:
    - ((start_node, end_node), path)
    """
    start_node, end_node = pair
    start_pos = node_pos_matrix[start_node]
    end_pos = node_pos_matrix[end_node]
    path = calculate_shortest_path(pgm_map_matrix, start_pos, end_pos)
    return (start_node, end_node), path

def precompute_all_paths():
    """
    预计算所有兴趣点之间的路径并存储在字典中。
    使用多进程来加速计算。
    """
    # 获取所有节点的组合（兴趣点对）
    nodes = list(range(nodes_num))
    node_pairs = list(itertools.combinations(nodes, 2))

    # 使用多进程池来加速计算
    precomputed_paths = {}
    with Pool(processes=cpu_count()) as pool:  # 使用可用的 CPU 核心数量
        # 使用 tqdm 进度条包装 map 函数
        for ((start_node, end_node), path) in tqdm(pool.imap(compute_path_for_pair, node_pairs), total=len(node_pairs), desc="Calculating paths", unit="pair"):
            # 将路径存储在字典中（双向路径都存储以便快速查找）
            precomputed_paths[(start_node, end_node)] = path
            precomputed_paths[(end_node, start_node)] = path

    return precomputed_paths

if __name__ == '__main__':
    paths_filename = f'./{map_name}_all_precomputed_paths.npy'

    # if os.path.isfile(paths_filename):
    #     precomputed_paths = np.load(paths_filename, allow_pickle=True).item()
    # else:
    precomputed_paths = precompute_all_paths()

