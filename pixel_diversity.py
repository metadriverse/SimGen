import os
import numpy as np
from PIL import Image
import argparse

def calculate_variance(images_directory):
    image_files = os.listdir(images_directory)
    image_data = []

    for image_file in image_files[::50]:
        print(image_file)
        image = Image.open(os.path.join(images_directory, image_file))
        image_arr = np.array(image)
        image_data.append(image_arr)

    data_list= np.stack(image_data)
    data_list = data_list[:,::2,::2,:]
    print(data_list.shape)
    data_list = data_list.reshape(-1, 3)/255.0

    pixels_var = np.std(data_list, axis=0)
    pixels_var = np.mean(pixels_var)

    return pixels_var

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate pixel diversity of images in a directory')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing images')
    args = parser.parse_args()

    pixels_variance = calculate_variance(args.input_dir)
    print(f"Pixel diversity (variance): {pixels_variance}")