

from lib import *

@dataclass
class DataPaths:
    """Data paths configuration"""
    anno_dir: Path
    image_dir: Path
    cropped_image_dir: Path
    augmented_dir: Path
    error_log: Path


@dataclass
class PlotBoundingBox:
    """Bounding box representation"""
    x: int
    y: int
    w: int
    h: int
