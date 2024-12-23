
from lib import *
from utils import *
from transformations import *
@dataclass
class Config:
    required_field: list[str] = field(
        default_factory=lambda: ['text_label', 'value', 'color'])
    not_valid_value: dict[str, list[str]] = field(default_factory=lambda: {
        'color': ['UNK', 'dark greenn']
    })
    resize_image_size: Tuple[int, int] = field(
        default_factory=lambda: (512, 512))
    transformation: dict[str, Callable] = field(default_factory=lambda: {
        'color': closest_color, #CHARTQA
        'bbox': scaling_bbox, #CHARTQA
        'text_bbox': scaling_bbox,#CHARTQA
        'figure_info_bbox': scaling_bbox, #CHARTQA
        'polygon': scaling_bbox #BENETECH
    })


# def extend_bbox(image: np.ndarray, bbox: Dict[str, int]):



