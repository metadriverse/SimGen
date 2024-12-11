import os.path

from simgen import SimGenPipeline
from PIL import Image
import numpy as np

def load_images():
    # Load the condition images
    depth_image = "assets/metadrive_conds/cond_depth.jpg"
    seg_image = "assets/metadrive_conds/cond_seg.jpg"

    depth_image = Image.open(depth_image).convert('RGB')
    seg_image = Image.open(seg_image).convert('RGB')

    depth_image = np.array(depth_image).astype(np.uint8)
    seg_image = np.array(seg_image).astype(np.uint8)

    return depth_image, seg_image

if __name__ == '__main__':

    pipeline = SimGenPipeline()
    # In the actual implementation, the images will be provided by the simulator
    depth_image, seg_image = load_images()

    output = pipeline(depth_image, seg_image,
                      prompt="An image of a sunny city street",
                      seed=2011,
                      num_inference_steps=100,
                      conddiff_strength=0.65,
                      )
    Image.fromarray(output.images[0]).save("output.jpg")
    print("Test done. Image saved at output.jpg")





