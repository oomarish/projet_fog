import numpy as np
import cv2
import requests
import concurrent.futures
import time
from flask import Flask, request, render_template, jsonify, send_from_directory
from scipy.ndimage import convolve
import os
from werkzeug.utils import secure_filename
import base64

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
slave_nodes = ['ip@1', 'ip@2', 'ip@3']

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to send image parts (color channel) to slaves
def send_to_slave(slave_url, channel, kernel):
    response = requests.post(f"{slave_url}/process", json={
        'image_part': channel.tolist(),
        'kernel': kernel.tolist()
    })
    return np.array(response.json()['processed_part'])

# Function to convert image to base64 for HTML rendering
def convert_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Main route to upload and process the image
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the image in color (BGR format)
        image = cv2.imread(filepath)

        # Split the image into B, G, R channels
        blue_channel, green_channel, red_channel = cv2.split(image)

        # Define a large convolution kernel
        kernel = np.ones((30, 30)) / 900  # A large 30x30 kernel

        # Start the timer
        start_time = time.time()

        # Send each color channel (R, G, B) to a different slave
        channels = [red_channel, green_channel, blue_channel]
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(send_to_slave, slave, channel, kernel) 
                       for slave, channel in zip(slave_nodes, channels)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # Recombine the processed R, G, B channels
        processed_image = cv2.merge((results[2], results[1], results[0]))  # Merge as BGR

        # Stop the timer and calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Time taken for distributed processing: {elapsed_time:.2f} seconds")

        # Convert original and processed images to base64 for display
        original_image_base64 = convert_to_base64(image)
        processed_image_base64 = convert_to_base64(processed_image)

        # Convert processed channel images to base64
        # Create 3-channel images for visualization
        processed_images_base64 = []
        for i, channel in enumerate(results):
            # Create a blank image with 3 channels
            color_channel_image = np.zeros_like(image)
            color_channel_image[..., i] = channel  # Assign the processed channel
            processed_images_base64.append(convert_to_base64(color_channel_image))

        # Render the result page with the elapsed time and processed images
        return render_template('result.html', 
                               elapsed_time=elapsed_time, 
                               original_image=original_image_base64, 
                               processed_image=processed_image_base64,
                               processed_images=processed_images_base64)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
