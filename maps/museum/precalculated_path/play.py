import numpy as np

# # 加载原始路径字典
paths = np.load('museum_all_precomputed_paths.npy', allow_pickle=True).item()
nodes = np.load('../museum_node_positions.npy')
#
# 创建一个新的字典来存储修正后的路径
corrected_paths = {}

# 遍历原始路径字典中的每一个键值对
for (start_node, end_node), path in paths.items():
    if start_node < end_node:
        # 如果 start_node < end_node，直接将路径添加到新字典中
        corrected_paths[(start_node, end_node)] = path
    else:
        try:
            # 如果 start_node > end_node，将路径翻转并添加到新字典中
            reverse_path = path[::-1]
            chop_head_cut_tail = reverse_path[1:]
            start = int(nodes[end_node][0])
            end = int(nodes[end_node][1])
            chop_head_cut_tail.append((start, end))
            corrected_paths[(start_node, end_node)] = chop_head_cut_tail
        except:
            corrected_paths[(start_node, end_node)] = path

# 将修正后的字典存储到新的文件中
np.save('../museum_corrected_paths.npy', corrected_paths)

# 测试输出，检查一些路径是否已经正确修正
print(corrected_paths[(0, 1)])
print()
print(corrected_paths[(1, 0)])
