from collections import deque
import numpy as np
#
# # Load precomputed paths
precomputed_paths = np.load('cumberland_corrected_paths.npy', allow_pickle=True).item()
map_adj_matrix = np.load('../cumberland_adj_matrix.npy')

def bfs_shortest_path(adj_matrix, start, goal):
    """Finds the shortest path in terms of node sequence using BFS."""
    visited = set()
    queue = deque([(start, [])])

    while queue:
        (vertex, path) = queue.popleft()
        if vertex in visited:
            continue

        visited.add(vertex)
        path = path + [vertex]

        if vertex == goal:
            return path

        for neighbor, connected in enumerate(adj_matrix[vertex]):
            if connected and neighbor not in visited:
                queue.append((neighbor, path))

    return None  # Return None if no path is found


def get_full_path(precomputed_paths, adj_matrix, start_node, goal_node):
    """Get the full path between two nodes, attempting to handle missing segments."""

    def find_path_between_nodes(current_node, goal_node):
        """Helper function to find path between current and goal node using BFS."""
        # Using BFS to find the shortest path in terms of node sequence
        node_path = bfs_shortest_path(adj_matrix, current_node, goal_node)
        return node_path

    # Start with an initial path search
    node_path = find_path_between_nodes(start_node, goal_node)

    if node_path is None:
        return None

    full_path = []  # To hold the complete coordinate path
    checked_pairs = set()  # To track checked node pairs to avoid loops

    i = 0
    while i < len(node_path) - 1:
        from_node = node_path[i]
        to_node = node_path[i + 1]

        # Skip already checked pairs
        if (from_node, to_node) in checked_pairs:
            i += 1
            continue

        path_segment = precomputed_paths.get((from_node, to_node))

        if path_segment:
            # If path segment exists, extend the full path
            full_path.extend(path_segment)
            i += 1  # Move to the next node in path
        else:
            # Mark this pair as checked
            checked_pairs.add((from_node, to_node))
            alternative_path = find_path_between_nodes(from_node, goal_node)

            if alternative_path is None or len(alternative_path) < 2:
                # No valid alternative path found; we are stuck
                return None

            # Set the new node path from the alternative path found
            node_path = node_path[:i + 1] + alternative_path[1:]  # Merge alternative path into current path

    return full_path


for i in range(40):
    for j in range(i+1,40):
        if get_full_path(precomputed_paths, map_adj_matrix,i,j):
            print(f'Node {i} Node{j} connected')
        else:
            print(f"Can't find between Node {i} Node{j}")

# full_path = get_full_path(precomputed_paths, map_adj_matrix, start_node, goal_node)
# if full_path:
#     print(f"Full path: {full_path}")
# else:
#     print("No path found.")


# import numpy as np
#
# # Load precomputed paths
# precomputed_paths = np.load('cumberland_corrected_paths.npy', allow_pickle=True).item()
#
#
# def check_all_connected_nodes_have_paths(adj_matrix, precomputed_paths):
#     """
#     Check if all connected nodes in an adjacency matrix have paths in the precomputed paths dictionary.
#     """
#     num_nodes = len(adj_matrix)
#     all_paths_exist = True
#
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if adj_matrix[i][j] >= 1:
#                 # Check if there is a precomputed path from i to j
#                 if (i, j) not in precomputed_paths or precomputed_paths[(i, j)] == False:
#                     print(f"No valid path exists between connected nodes {i} and {j} in precomputed paths.")
#                     all_paths_exist = False
#                 else:
#                     print(f"Valid path exists between connected nodes {i} and {j}.")
#
#     return all_paths_exist
#
# # Run the check
# all_paths_exist = check_all_connected_nodes_have_paths(map_adj_matrix, precomputed_paths)
#
# if all_paths_exist:
#     print("All connected nodes in the adjacency matrix have valid paths in the precomputed paths.")
# else:
#     print("Some connected nodes in the adjacency matrix do not have valid paths in the precomputed paths.")
