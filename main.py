import cv2
import numpy as np

def reduce_intensity_levels(image_path, num_levels):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read the image in grayscale mode
    max_intensity = 255  # Define the maximum intensity level
    new_intensity = max_intensity // (num_levels - 1) # Calculate the new intensity level based on the number of desired levels
    quantized_image = np.floor_divide(image, new_intensity) * new_intensity # Quantize the image by reducing intensity levels
    
    return quantized_image

input_image_path = "images/image.jpg" # Path to the input image
num_levels = 2 # Number of intensity levels to reduce the image to

reduced_image = reduce_intensity_levels(input_image_path, num_levels)

cv2.imwrite("images/reduced_image/reduced_image.jpg", reduced_image) # Save the reduced image