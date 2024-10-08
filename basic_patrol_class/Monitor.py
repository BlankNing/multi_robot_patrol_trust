import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image, ImageDraw

class Monitor():
    '''
    For recording metric behaviour and writing log
    '''
    def __init__(self):
        self.robot_pos = []
        self.node_idleness = []

    def collect_robot_pos(self, robots_pos_record):
        self.robot_pos.append(robots_pos_record)

    def collect_node_idleness(self, nodes_idleness_record):
        self.node_idleness.append(nodes_idleness_record)

    def plot_idleness(self, node_num):
        data_array = np.array(self.node_idleness)
        column_data = data_array[:, node_num]
        plt.plot(range(len(column_data)), column_data)
        plt.title(f"Idleness of Node {node_num} Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Idleness")
        plt.grid(True)
        plt.show()

    def plot_all_idleness(self):
        data_array = np.array(self.node_idleness)  # Convert node idleness data to a numpy array
        num_nodes = data_array.shape[1]  # Get the number of nodes

        plt.figure(figsize=(10, 6))  # Set the figure size

        for node_num in range(num_nodes):  # Iterate over each node
            column_data = data_array[:, node_num]  # Get the idleness data for each node
            plt.plot(range(len(column_data)), column_data,
                     label=f"Node {node_num}")  # Plot the idleness curve for the node

        plt.title("Idleness of All Nodes Over Time")  # Set the plot title
        plt.xlabel("Time Step")  # Set the x-axis label
        plt.ylabel("Idleness")  # Set the y-axis label
        plt.legend(loc="upper right")  # Add a legend, placed in the upper right corner
        plt.grid(True)  # Show grid lines
        plt.show()  # Display the plot

    def plot_idleness_in_range(self, node_range):
        data_array = np.array(self.node_idleness)  # Convert node idleness data to a numpy array

        plt.figure(figsize=(10, 6))  # Set the figure size

        for node_num in node_range:  # Iterate over the specified range of nodes
            column_data = data_array[:, node_num]  # Get the idleness data for each node
            plt.plot(range(len(column_data)), column_data,
                     label=f"Node {node_num}")  # Plot the idleness curve for the node

        plt.title("Idleness of Nodes in Specified Range Over Time")  # Set the plot title
        plt.xlabel("Time Step")  # Set the x-axis label
        plt.ylabel("Idleness")  # Set the y-axis label
        plt.legend(loc="upper right")  # Add a legend, placed in the upper right corner
        plt.grid(True)  # Show grid lines
        plt.show()  # Display the plot

    def create_patrol_screenshot(self, config_file, time_step, save_path=None, save_plot=False):
        node_pos_matrix = config_file['env_config']['node_pos_matrix']
        pgm_map_matrix = config_file['env_config']['pgm_map_matrix']
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [15, 1]}, figsize=(11, 6))

        ax1.imshow(pgm_map_matrix, cmap='gray')

        ax1.scatter(node_pos_matrix[:, 0], node_pos_matrix[:, 1], color='red')

        for i, pos in enumerate(node_pos_matrix):
            ax1.text(pos[0], pos[1], str(i), color='blue', fontsize=8, ha='center', va='center')

        robot_positions = self.robot_pos[time_step]
        colors = cm.rainbow(np.linspace(0, 1, len(robot_positions)))
        for i, positions in enumerate(robot_positions):
            x = positions[0]
            y = positions[1]
            ax1.scatter(x, y, color=colors[i], label=f"Robot {i}")

        ax2.axis('off')
        for i, positions in enumerate(robot_positions):
            ax2.scatter([], [], color=colors[i], label=f"Robot {i}")
        ax2.legend(loc='center right')
        if save_plot:
            plt.savefig(save_path)
        else:
            plt.show()


    def create_patrol_gif(self, config_file, output_filename='patrol.gif'):
        print("Start Creating GIF animation")
        node_pos_matrix = config_file['env_config']['node_pos_matrix']
        pgm_map_matrix = config_file['env_config']['pgm_map_matrix']
        robots_num = config_file['robot_config']['robots_num']
        height, width = pgm_map_matrix.shape[0], pgm_map_matrix.shape[1]

        # Create the background image
        background = Image.new('RGB', (width, height), 'white')
        draw_bg = ImageDraw.Draw(background)

        # Draw the static background (walls and open space)
        for y in range(height):
            for x in range(width):
                if pgm_map_matrix[y][x] == 0:
                    draw_bg.point((x, y), fill='black')

        # Draw interest points
        def draw_5x5_square(draw, position, color):
            x, y = position
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    draw.point((x + dx, y + dy), fill=color)

        def draw_9x9_square_np(draw, position, color):
            x = position[0]
            y = position[1]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    draw.point((x + dx, y + dy), fill=color)

        for position in node_pos_matrix:
            draw_9x9_square_np(draw_bg, position, (255, 0, 0))

        frames = []
        for t in tqdm(range(len(self.robot_pos)), desc="Creating GIF frames", unit="frame"):
            img = background.copy()
            draw = ImageDraw.Draw(img)

            # Generate an infinite number of colors using cm.rainbow
            num_colors = robots_num
            colors = [tuple((np.array(cm.rainbow(i / num_colors)[:3]) * 255).astype(int)) for i in range(num_colors)]

            for i in range(robots_num):
                robot_i_pos = np.array(self.robot_pos)[:, i]
                color = colors[i % len(colors)]
                interval = 2
                start_step = max(0, t * interval - 9)  # Ensure we don't go below 0
                for position in robot_i_pos[start_step:t * interval]:
                    draw_5x5_square(draw, position, color)
            frames.append(img)
        frames[0].save(output_filename, save_all=True, append_images=frames[1:], loop=0, duration=20)

    def create_patrol_gif_new(self, config_file, output_filename='patrol.gif'):
        print("Start Creating GIF animation")
        node_pos_matrix = config_file['env_config']['node_pos_matrix']
        pgm_map_matrix = config_file['env_config']['pgm_map_matrix']
        robots_num = config_file['robot_config']['robots_num']
        height, width = pgm_map_matrix.shape
        robot_positions = np.array(self.robot_pos)

        # Create the background image
        background = Image.new('RGB', (width, height), 'white')
        draw_bg = ImageDraw.Draw(background)

        # Draw the static background (walls and open space)
        wall_coords = np.argwhere(pgm_map_matrix == 0)
        for (y, x) in wall_coords:
            draw_bg.point((x, y), fill='black')

        # Draw interest points
        def draw_square(draw, position, color, size=4):
            x, y = position
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    draw.point((x + dx, y + dy), fill=color)

        for position in node_pos_matrix:
            draw_square(draw_bg, position, (255, 0, 0), size=4)

        # Pre-calculate colors for all robots
        colors = [tuple((np.array(cm.rainbow(i / robots_num)[:3]) * 255).astype(int)) for i in range(robots_num)]

        frames = []
        interval = 2
        frame_range = range(len(self.robot_pos))

        for t in tqdm(frame_range, desc="Creating GIF frames", unit="frame"):
            img = background.copy()
            draw = ImageDraw.Draw(img)

            for i in range(robots_num):
                robot_i_pos = robot_positions[:, i]
                color = colors[i]

                start_step = max(0, t * interval - 9)  # Ensure we don't go below 0
                for position in robot_i_pos[start_step:t * interval]:
                    draw_square(draw, position, color, size=2)

            frames.append(img)

        # Save the GIF
        frames[0].save(output_filename, save_all=True, append_images=frames[1:], loop=0, duration=20)