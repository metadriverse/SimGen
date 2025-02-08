from PIL import Image
from typing import Tuple


def get_aspect_ratio(
    image: Image.Image | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> Tuple[int, int]:
    """ Check the aspect ratio of the input image or provided width and height """
    if isinstance(image, Image.Image):
        if image_width is None:
            image_width = image.width
        if image_height is None:
            image_height = image.height
    if image is not None and image_width is None and image_height is None:
        raise ValueError(
            "input format of method \'get_aspect_ratio\' is incorrect. Please provide either image or image_width and image_height"
        )
    if image is not None:
        image_width, image_height = image.size
    aspect_ratio = image_width / image_height
    return aspect_ratio
