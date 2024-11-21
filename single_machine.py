import numpy as np
import cv2
import time
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from tqdm import tqdm  # For progress bar

# Load a high-resolution image
image = cv2.imread('img.jpg')  # Ensure the image path is correct
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display in matplotlib

# Split the image into R, G, B channels
red_channel, green_channel, blue_channel = cv2.split(image)

# Define a large convolution kernel
kernel = np.ones((30, 30)) / 900  # A large 30x30 kernel

# Start the timer
start_time = time.time()

# Apply convolution multiple times for complexity on each channel
for _ in tqdm(range(20), desc="Applying convolutions"):  # Adding tqdm progress bar
    red_channel = convolve(red_channel, kernel)
    green_channel = convolve(green_channel, kernel)
    blue_channel = convolve(blue_channel, kernel)

# Merge the processed R, G, B channels
processed_image = cv2.merge((red_channel, green_channel, blue_channel))

# Stop the timer and calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Time taken for single-machine processing: {elapsed_time:.2f} seconds")

# Display the final processed image
plt.imshow(processed_image.astype(np.uint8))  # Ensure correct type for display
plt.title('Processed Image with Multiple Convolutions')
plt.show()
