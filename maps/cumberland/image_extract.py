import cv2
import numpy as np

# Load the image
image = cv2.imread('points_of_cumberland.pgm', cv2.IMREAD_GRAYSCALE)  # Replace 'image.png' with your image path

# Find the coordinates of non-zero pixels
non_zero_coords = np.transpose(np.nonzero(image))

print(non_zero_coords)

