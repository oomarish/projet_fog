from flask import Flask, request, jsonify
import numpy as np
from scipy.ndimage import convolve

app = Flask(__name__)

# Number of convolution passes (matching single-machine complexity)
NUM_PASSES = 20

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    image_part = np.array(data['image_part'])
    kernel = np.array(data['kernel'])

    # Apply the convolution multiple times (100 passes)
    for _ in range(NUM_PASSES):
        image_part = convolve(image_part, kernel)

    # Return the processed part as JSON
    return jsonify({'processed_part': image_part.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
