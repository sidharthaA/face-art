from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def cartoonize_image(img):
    # Apply cartoon effect to the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
def create_pop_art(img, max_dots=120, multiplier=100, output_size=None):
    # Load the image as grayscale
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image based on the maximum number of dots
    original_height, original_width = original_image.shape
    if original_height > original_width:
        new_width = int(original_width * (max_dots / original_height))
        new_height = max_dots
    else:
        new_width = max_dots
        new_height = int(original_height * (max_dots / original_width))
    downsized_image = cv2.resize(original_image, (new_width, new_height))

    # Create a blank canvas
    blank_image = np.full((new_height * multiplier, new_width * multiplier, 3), [19, 247, 224], dtype=np.uint8)

    # Draw circles on the blank canvas
    padding = int(multiplier / 2)
    for y in range(new_height):
        for x in range(new_width):
            radius = int(0.6 * multiplier * ((255 - downsized_image[y, x]) / 255))
            cv2.circle(blank_image, ((x * multiplier + padding), (y * multiplier + padding)), radius, (247, 19, 217), -1)
    if output_size:
        final_width, final_height = output_size
        blank_image = cv2.resize(blank_image, (final_width, final_height))

    # Save the resulting image
    # cv2.imwrite(output_path, blank_image)
    return blank_image


def mosaic_image(img, w=100, h=100):
    # Apply mosaic effect to the image
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (img.shape[1]*5, img.shape[0]*5), interpolation=cv2.INTER_NEAREST)
    return img

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filter_type = request.form['filter']
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    img = cv2.imdecode(np.frombuffer(in_memory_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if filter_type == 'cartoon':
        result = cartoonize_image(img)
    elif filter_type == 'mosaic':
        result = mosaic_image(img)
    elif filter_type == 'pop_art':
        result = create_pop_art(img, output_size=(500, 500))

    _, buffer = cv2.imencode('.png', result)
    img_str = base64.b64encode(buffer).decode()
    return jsonify({'image': 'data:image/png;base64,' + img_str})

    # result_encoded = cv2.imencode('.png', result)[1].tobytes()
    # return send_file(BytesIO(result_encoded), download_name='filtered.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
