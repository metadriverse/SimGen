from flask import Flask, send_from_directory
import os
import argparse
import time

app = Flask(__name__)

def create_app(image_folder, condition_num=4, indices=None):
    if indices is None:
        indices = {}
    
    @app.route('/')
    def show_index():
        images = sorted(os.listdir(image_folder))
        images = [images[i] for i in range(len(images)) if i not in indices]
        html = "<html><body>"
        count = 0
        for image in images:
            count += 1
            if count % (condition_num+3) == 1:
                html += "<img src='/" + image + "' style='width:100%; height:25px; object-fit: cover; object-position: top;'>"
            else:
                html += "<img src='/" + image + "' style='width:100%; height:auto;'>"
            if count % (condition_num+3) == 0: 
                html += "<h2 style='text-align: center;'>{}</h2><hr>".format(image) 
        html += "</body></html>"
        return html

    @app.route('/<filename>')
    def send_image(filename):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return send_from_directory(image_folder, filename)
            except Exception as e:
                print(f"Failed to load image: {e}")
                retries += 1
                time.sleep(RETRY_DELAY)
        return "Failed to load image"

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web server for displaying images')
    parser.add_argument('--image-folder', type=str, required=True, help='Directory containing images')
    parser.add_argument('--condition-num', type=int, default=4, help='Number of conditions')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()

    MAX_RETRIES = 10
    RETRY_DELAY = 1

    app = create_app(args.image_folder, args.condition_num)
    app.run(debug=False, port=args.port)
