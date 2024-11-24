
from lib import *
def scaling_bbox(bbox, **kwargs):
    """Scaling the bbox annotation from the original size to desired size

    Args:
        bbox: The dictionary containing x, y, w, h values
        **kwargs: Additional parameters including:
            - original_size: tuple of (H, W) or dict containing original_size and desired_size
            - desired_size: The desired size of the image (H, W)
    """
    original_size = kwargs.get('original_size')
    desired_size = kwargs.get('desired_size')

    if isinstance(original_size, dict):
        desired_size = original_size['desired_size']
        original_size = original_size['original_size']

    if not all([original_size, desired_size, bbox]):
        raise ValueError(
            "Missing required parameters: original_size, desired_size, or bbox")

    scaling_x = desired_size[1] / original_size[1]
    scaling_y = desired_size[0] / original_size[0]

    bbox_res = {}
    bbox_res['x'] = bbox['x'] * scaling_x
    bbox_res['y'] = bbox['y'] * scaling_y
    bbox_res['w'] = bbox['w'] * scaling_x
    bbox_res['h'] = bbox['h'] * scaling_y

    return bbox_res


