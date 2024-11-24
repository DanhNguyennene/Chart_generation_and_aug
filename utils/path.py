

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
    """Bounding box representation
    STRUCT:
        x: x-axis
        y: y-axis
        w: w-axis
        h: h-axis
    """
    x: int
    y: int
    w: int
    h: int
