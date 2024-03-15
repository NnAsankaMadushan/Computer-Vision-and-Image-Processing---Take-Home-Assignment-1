import cv2
import numpy as np

def reduce_intensity_levels(image_path, value):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read the image in grayscale mode
    new_intensity = 2 ** (8-value) # Calculate the new intensity level based on the number of desired levels
    quantized_image = np.floor_divide(image, new_intensity) * new_intensity # Quantize the image by reducing intensity levels
    
    return quantized_image



input_image_path = "images/image.jpg" # Path to the input image
value = 3 # Number of intensity levels to reduce the image to

reduced_image = reduce_intensity_levels(input_image_path, value)

cv2.imwrite("images/1-0_reduced_image.jpg", reduced_image) # Save the reduced image
# Print a success message
print("The image intensity has been reduced and saved as 1-0_reduced_image.jpg")

#..................................................................................................................................................................................

def spatial_average(image_path, neighborhood_size):
    image = cv2.imread(image_path)
    blurred_image = cv2.blur(image, (neighborhood_size, neighborhood_size))

    return blurred_image

input_image_path = 'images/image.jpg'
neighborhood_3x3 = 3
averaged_image_3x3 = spatial_average(input_image_path, neighborhood_3x3)
cv2.imwrite("images/2-1_averaged_image_3x3.jpg", averaged_image_3x3)
# Print a success message
print("The image has been saved as 2-1_averaged_image_3x3.jpg")

neighborhood_10x10 = 10
averaged_image_10x10 = spatial_average(input_image_path, neighborhood_10x10)
cv2.imwrite("images/2-2_averaged_image_10x10.jpg", averaged_image_10x10)
# Print a success message
print("The image has been saved as 2-2_averaged_image_10x10.jpg")

neighborhood_20x20 = 20
averaged_image_20x20 = spatial_average(input_image_path, neighborhood_20x20)
cv2.imwrite("images/2-3_averaged_image_20x20.jpg", averaged_image_20x20)
# Print a success message
print("The image has been saved as 2-3_averaged_image_20x20.jpg")

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
cv2.imwrite('images/3-1_rotated_image_45.jpg', rotated_image_45)
# Print a success message
print("The 45 rotated image has been saved as 3-1_rotated_image_45.jpg")

cv2.imwrite('images/3-2_rotated_image_90.jpg', rotated_image_90)
# Print a success message
print("The 90 rotated image has been saved as 3-2_rotated_image_45.jpg")

#..................................................................................................................................................................................

def block_average(image, block_size):
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    averaged_image = np.zeros_like(image, dtype=np.float64)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            average_value = np.mean(block, axis=(0, 1))
            averaged_image[y:y+block_size, x:x+block_size] = average_value

    return np.uint8(averaged_image)
# Example usage:
# input_image_path = 'images/image.jpg'
image = cv2.imread('images/image.jpg', cv2.IMREAD_COLOR)
block_size_3x3 = 3
averaged_image_3x3_blocks = block_average(image, block_size_3x3)
cv2.imwrite('images/averaged_image_3x3_blocks.jpg', averaged_image_3x3_blocks)

block_size_5x5 = 5
averaged_image_5x5_blocks = block_average(image, block_size_5x5)
cv2.imwrite('images/averaged_image_5x5_blocks.jpg', averaged_image_5x5_blocks)

block_size_7x7 = 7
averaged_image_7x7_blocks = block_average(image, block_size_7x7)
cv2.imwrite('images/averaged_image_7x7_blocks.jpg', averaged_image_7x7_blocks)