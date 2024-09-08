import numpy as np

# # # 加载原始路径字典
# paths = np.load('cumberland_all_precomputed_paths.npy', allow_pickle=True).item()
nodes = np.load('../cumberland_node_positions.npy')
# #
# # 创建一个新的字典来存储修正后的路径
# corrected_paths = {}
#
# # 遍历原始路径字典中的每一个键值对
# for (start_node, end_node), path in paths.items():
#     if start_node < end_node:
#         # 如果 start_node < end_node，直接将路径添加到新字典中
#         corrected_paths[(start_node, end_node)] = path
#     else:
#         try:
#             # 如果 start_node > end_node，将路径翻转并添加到新字典中
#             reverse_path = path[::-1]
#             chop_head_cut_tail = reverse_path[1:]
#             start = int(nodes[end_node][0])
#             end = int(nodes[end_node][1])
#             chop_head_cut_tail.append((start, end))
#             corrected_paths[(start_node, end_node)] = chop_head_cut_tail
#         except:
#             corrected_paths[(start_node, end_node)] = path
#
# # 将修正后的字典存储到新的文件中
# np.save('cumberland_corrected_paths.npy', corrected_paths)
#
# # 测试输出，检查一些路径是否已经正确修正
# print(corrected_paths[(0, 1)])
# print()
# print(corrected_paths[(1, 0)])


import pandas as pd
import numpy as np

# 读取数据
file_name = "list_of_lists.pkl"
objects = pd.read_pickle(file_name)
adj = np.load('../cumberland_adj_matrix.npy')
#
# def check_consistency(adj, objects):
#
#     # 遍历邻接矩阵
#     for i in range(adj.shape[0]):
#         for j in range(adj.shape[1]):
#             if adj[i, j] != 0:
#                 # 如果邻接矩阵中的元素不为0，检查objects中对应位置是否也不为0
#                 if objects[i][j] == 0:
#                     print(f"不一致: adj[{i}, {j}] = {adj[i, j]}, 但 objects[{i}][{j}] = 0")
#                     return False
#
#     print("检查通过：adj中不为0的位置在objects中对应位置也不为0")
#     return True
#
#
# # 运行检查
# result = check_consistency(adj, objects)
# print("最终结果:", result)


path = {}

for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        if objects[i][j] != 0:
            path[(i, j)] = objects[i][j]
        else:
            path[(i, j)] = False

np.save('cumberland_neighbour_precomputed_paths.npy', path, allow_pickle=True)
print(path[(0,2)])
print(path[(2,0)])
