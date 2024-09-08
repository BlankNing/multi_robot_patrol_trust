from PIL import Image
import numpy as np

def read_pgm_image(file_path):
    with open(file_path, 'rb') as f:
        img = Image.open(f)
        img_array = np.array(img)
    return img_array

def get_map_adj_matrix(map_name):
    return np.load(f'./maps/{map_name}/{map_name}_adj_matrix.npy')

def get_node_pos_matrix(map_name):
    return np.load(f'./maps/{map_name}/{map_name}_node_positions.npy')

def get_pgm_map_matrix(map_name):
    return read_pgm_image(f'./maps/{map_name}/{map_name}.pgm')

def get_default_init_pos(node_pos_matrix, robots_num):
    if len(node_pos_matrix) < robots_num:
        raise ValueError("The number of nodes in node_pos_matrix must be at least equal to robots_num.")

    step = len(node_pos_matrix) / robots_num
    indices = [int(i * step) for i in range(robots_num)]

    indices = [min(i, len(node_pos_matrix) - 1) for i in indices]

    return [tuple(node_pos_matrix[i]) for i in indices]

def get_predefined_path(map_name):
    try:
        pre_path = np.load(f'./maps/{map_name}/{map_name}_corrected_paths.npy', allow_pickle=True).item()
    except:
        pre_path = None
    return pre_path