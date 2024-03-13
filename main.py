import cv2
import numpy as np

def reduce_intensity_levels(image_path, num_levels):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read the image in grayscale mode
    max_intensity = 255  # Define the maximum intensity level
    new_intensity = max_intensity // (num_levels - 1) # Calculate the new intensity level based on the number of desired levels
    quantized_image = np.floor_divide(image, new_intensity) * new_intensity # Quantize the image by reducing intensity levels
    
    return quantized_image

#..................................................................................................................................................................................

input_image_path = "images/image.jpg" # Path to the input image
num_levels = 2 # Number of intensity levels to reduce the image to

reduced_image = reduce_intensity_levels(input_image_path, num_levels)

cv2.imwrite("images/reduced_image.jpg", reduced_image) # Save the reduced image

def spatial_average(image_path, neighborhood_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.blur(image, (neighborhood_size, neighborhood_size))

    return blurred_image

input_image_path = 'images/image.jpg'
neighborhood_3x3 = 3
averaged_image_3x3 = spatial_average(input_image_path, neighborhood_3x3)
cv2.imwrite("images/averaged_image_3x3.jpg", averaged_image_3x3)

#..................................................................................................................................................................................

def rotate_image(image_path, angle_degrees):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

input_image_path = 'images/image.jpg'
rotated_image_45 = rotate_image(input_image_path, 45)
rotated_image_90 = rotate_image(input_image_path, 90)
cv2.imwrite('images/rotated_image_45.jpg', rotated_image_45)
cv2.imwrite('images/rotated_image_90.jpg', rotated_image_90)

#..................................................................................................................................................................................

def block_average(image_path, block_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]
    averaged_image = np.zeros_like(image)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            average_value = np.mean(block)
            averaged_image[y:y+block_size, x:x+block_size] = average_value

    return averaged_image
# Example usage:
input_image_path = 'images/image.jpg'
block_size_3x3 = 3
averaged_image_3x3_blocks = block_average(input_image_path, block_size_3x3)
cv2.imwrite('images/averaged_image_3x3_blocks.jpg', averaged_image_3x3_blocks)