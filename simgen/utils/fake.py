import requests
from PIL import Image

def get_a_image(
        image_width: int | None = None,
        image_height: int | None = None,
) -> Image.Image:
    # prepare image and text prompt
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)


    if image_width is not None and image_height is not None:
        # We first resize the longest side to the target size
        width, height = image.size
        if width > height:
            image = image.resize((image_width, int(image_width * height / width)))
        else:
            image = image.resize((int(image_height * width / height), image_height))
        # Then we crop the image to the target size
        width, height = image.size
        left = (width - image_width) / 2
        top = (height - image_height) / 2
        right = (width + image_width) / 2
        bottom = (height + image_height) / 2
        image = image.crop((left, top, right, bottom))

    return image