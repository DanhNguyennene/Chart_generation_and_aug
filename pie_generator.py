import os
import json
from joblib import Parallel, delayed
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from tqdm import tqdm
from typing import Dict, Tuple, List, Any, Optional, Set, Union, Callable
from copy import deepcopy
from pprint import pprint
from dataclasses import dataclass, field
import matplotlib
from matplotlib.colors import to_rgb, to_hex
from scipy.spatial import distance
import cv2
import numpy as np
import logging
from pathlib import Path


def generate_random_color_bgr() -> Tuple[int, int, int]:
    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)
    return blue, green, red


# CONSTANT


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


class ChartAugmentor:
    def __init__(self, paths: DataPaths):
        self.paths = paths
        self._setup_logging()
        self._setup_directories()

    def _setup_logging(self) -> None:
        """Configure logging"""
        logging.basicConfig(
            filename=self.paths.error_log,
            level=logging.ERROR,
            format='%(asctime)s - %(message)s'
        )

    def _setup_directories(self) -> None:
        """Ensure all required directories exist"""
        self.paths.augmented_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_image(path: Path) -> Optional[np.ndarray]:
        """Load and convert image to RGB"""
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}")
            return None

    @staticmethod
    def save_image(image: np.ndarray, save_path: Path) -> bool:
        """Save image in BGR format"""
        try:
            cv2.imwrite(str(save_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            logging.error(f"Error saving image {save_path}: {str(e)}")
            return False

    @staticmethod
    def extend_bbox(image: np.ndarray, bbox: PlotBoundingBox) -> PlotBoundingBox:
        """Extend bounding box until white pixels are encountered"""
        height, width = image.shape[:2]

        # Extend downward
        h = bbox.h
        while (bbox.y + h < height - 1 and
               np.any(image[(bbox.y + h):(bbox.y + h + 1), bbox.x:bbox.x + bbox.w] != [255, 255, 255])):
            h += 1

        return PlotBoundingBox(bbox.x, bbox.y, bbox.w, h)

    def mask_image(self, base_image: np.ndarray, overlay_image: np.ndarray, original_bbox: PlotBoundingBox) -> Optional[np.ndarray]:
        """
        Overlay image at a random position within the original bounding box while maintaining aspect ratio

        Args:
            base_image: The base image to augment
            overlay_image: The image to overlay
            bbox: The current bounding box
            original_bbox: The original bounding box to stay within

        Returns:
            Augmented image or None if operation fails
        """
        try:
            resized_overlay = cv2.resize(overlay_image, (original_bbox.w, original_bbox.h))

            result = base_image.copy()  
            result[original_bbox.y:original_bbox.y + original_bbox.h,
                   original_bbox.x:original_bbox.x + original_bbox.w] = resized_overlay

            return result

        except Exception as e:
            logging.error(f"Error in mask_image: {str(e)}")
            return None

    def save_json(self, data: Dict, save_path: Path) -> bool:
        """Save JSON data to file"""
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Error saving JSON to {save_path}: {str(e)}")
            return False

    def process_single_chart(self, anno_path: Path, num_augmentations: int = 10) -> None:
        """
        Process a single chart annotation file with random cropped image selection

        Args:
            anno_path: Path to the annotation file
            num_augmentations: Number of augmented images to generate
        """
        try:
            # Load annotation
            with open(anno_path) as f:
                data = json.load(f)

            if data['type'] != 'pie':
                return

            # Get base image
            image_path = self.paths.image_dir / \
                anno_path.with_suffix('.png').name
            base_image = self.load_image(image_path)
            if base_image is None:
                logging.error(f"Could not load base image: {image_path}")
                return

            # Get list of all cropped images
            crop_paths = list(self.paths.cropped_image_dir.glob('*.png'))
            if not crop_paths:
                logging.error(
                    f"No cropped images found in {self.paths.cropped_image_dir}")
                return
            sorted(crop_paths)
            # Extract and extend bbox
            bbox_data = data['general_figure_info']['figure_info']['bbox']
            bbox = PlotBoundingBox(**bbox_data)
            original_bbox = bbox
            bbox = self.extend_bbox(base_image, bbox)

            # Generate augmentations with different cropped images
            if num_augmentations == -1:
                num_augmentations = len(crop_paths)
            for i in range(num_augmentations):
                # Randomly select a cropped image
                crop_path = np.random.choice(crop_paths)
                cropped_image = self.load_image(crop_path)

                if cropped_image is None:
                    logging.warning(
                        f"Could not load cropped image: {crop_path}")
                    continue

                # Apply augmentation
                augmented = self.mask_image(
                    base_image, cropped_image, bbox, original_bbox)
                if augmented is not None:
                    out_path = self.paths.augmented_dir / \
                        f"{anno_path.stem}_aug{i}_{crop_path.stem}.png"
                    self.save_image(augmented, out_path)
                    self.save_json(data, self.paths.augmented_dir /
                                   f"{anno_path.stem}_aug{i}_{crop_path.stem}.json")
                else:
                    logging.warning(
                        f"Failed to create augmentation {i} with crop {crop_path}")

        except Exception as e:
            logging.error(f"Error processing {anno_path}: {str(e)}")

    def process_dataset(self, num_augmentations: int = 10) -> None:
        """Process all chart annotations in the dataset"""
        for anno_path in self.paths.anno_dir.glob('*.json'):
            self.process_single_chart(anno_path, num_augmentations)

    # def crop_single_image(self, image_path: Path) -> None:
    #     """Crop a single image to remove white space"""
    #     try:
    #         # Load image
    #         image = self.load_image(image_path)
    #         if image is None:
    #             logging.error(f"Could not load image: {image_path}")
    #             return

    #         # Convert to grayscale
    #         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #         # Find bounding box
    #         coords = cv2.findNonZero(gray)
    #         x, y, w, h = cv2.boundingRect(coords)

    #         # Crop image
    #         cropped = image[y:y+h, x:x+w]

    #         # Save cropped image
    #         save_path = self.paths.cropped_image_dir / image_path.name
    #         self.save_image(cropped, save_path)

    #     except Exception as e:
    #         logging.error(f"Error cropping {image_path}: {str(e)}")

    # def crop_all_images(self) -> None:
    #     """Crop all images in the dataset"""
    #     for image_path in self.paths.image_dir.glob('*.png'):
    #         self.crop_single_image(image_path)


def visualize_image(image: np.ndarray, json_dict: dict) -> np.ndarray:
    """Visualize the image

    Args:
        image (np.ndarray): the np array image
        json_dict (dict): the annotation

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: the annotated image
    """
    image_ann = image.copy()

    general_figure_info = json_dict.get('general_figure_info', None)
    if general_figure_info:
        figure_info_bbox = general_figure_info['figure_info']['bbox']
        title_bbox = general_figure_info['title']['bbox']

        x, y, w, h = map(int, [figure_info_bbox['x'], figure_info_bbox['y'],
                         figure_info_bbox['w'], figure_info_bbox['h']])
        cv2.rectangle(image_ann, (x, y), (x+w, y+h),
                      generate_random_color_bgr(), 2)
        x, y, w, h = map(
            int, [title_bbox['x'], title_bbox['y'], title_bbox['w'], title_bbox['h']])
        cv2.rectangle(image_ann, (x, y), (x+w, y+h),
                      generate_random_color_bgr(), 2)

    models = json_dict.get('models')
    if models:
        for model in models:
            text_bbox = model['text_bbox']
            x, y, w, h = map(
                int, [text_bbox['x'], text_bbox['y'], text_bbox['w'], text_bbox['h']])
            cv2.rectangle(image_ann, (x, y), (x+w, y+h),
                          generate_random_color_bgr(), 2)

    return image_ann


PALETTE = {name: to_rgb(hex)
           for name, hex in matplotlib.colors.TABLEAU_COLORS.items()}

SPECIAL_MAPPING = {
    'lightbrow': "#C4A484",
    'lightbrown': "#C4A484",
    "darkbrown": "#C4A484",
    "brow": "#964B00"
}

IMAGE_ERROR = set()


def closest_color(input_color: str) -> str:
    """
    Find the closest color from the palette to the input color.
    Args:
        input_color: Color as a name, hex, or RGB tuple.
    Returns:
        (closest_color_name, closest_color_hex)
    """
    global IMAGE_ERROR
    if input_color.split():
        input_color = ''.join(input_color.split())

    try:
        input_rgb = to_rgb(input_color)
    except Exception as e:
        return SPECIAL_MAPPING[input_color]
        # IMAGE_ERROR.add(input_color)
        # raise e
    distances = {name: distance.euclidean(
        input_rgb, rgb) for name, rgb in PALETTE.items()}

    closest_name = min(distances, key=distances.get)
    closest_hex = to_hex(PALETTE[closest_name])
    return closest_hex


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
        'color': closest_color,
        'bbox': scaling_bbox,
        'text_bbox': scaling_bbox,
        'figure_info_bbox': scaling_bbox,
    })

# def extend_bbox(image: np.ndarray, bbox: Dict[str, int]):


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


def print_transformed_values(json_data: dict, field: str, seen_values: set = None) -> set:
    """
    Recursively traverse JSON data and print unique transformations for a specific field.

    Args:
        json_data: The JSON data to analyze
        field: The field name to track transformations (e.g., 'color')
        seen_values: Set to track unique value pairs

    Returns:
        Set of tuples containing (original_value, transformed_value)
    """
    if seen_values is None:
        seen_values = set()

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == field:
                if isinstance(value, list):
                    for val in value:
                        try:
                            seen_values.add(val)
                        except ValueError:
                            continue
                else:
                    try:
                        seen_values.add(value)
                    except ValueError:
                        continue
            elif isinstance(value, (dict, list)):
                print_transformed_values(value, field, seen_values)
    elif isinstance(json_data, list):
        for item in json_data:
            print_transformed_values(item, field, seen_values)

    return seen_values


def extract_structure(data: dict) -> dict:
    """
    Extracts the structure of a JSON object by replacing values with placeholders.
    """
    if isinstance(data, dict):
        return {key: extract_structure(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [extract_structure(data[0])] if data else []
    else:
        return "value"


def find_unique_structures(json_list: list[dict], json_info_full) -> Dict[str, List[dict]]:
    """Read through each json, find the unique structure, and assign each structure, with a list of json that is represented by that structure

    Args:
        json_list (list[dict]): List of annotation
        json_info_full (_type_): The list of full annotation, including the full path and filename

    Returns:
        Dict[str, List[dict]]: return the mapping from a specific structure to the list of annotations(in full form, with filepath and filanem)
    """
    unique_structures_info = defaultdict(list)

    for i, json_data in tqdm(enumerate(json_list)):
        structure = extract_structure(json_data)

        structure_str = json.dumps(structure, sort_keys=True)

        unique_structures_info[structure_str].append(json_info_full[i])

    return len(unique_structures_info), unique_structures_info


def extract_keys(dictionary: dict) -> Set[str]:
    keys = set()

    def traverse(d: dict):
        for key, value in d.items():
            keys.add(key)
            if isinstance(value, dict):
                traverse(value)
            elif isinstance(value, list):
                for val in value:
                    if isinstance(val, dict):
                        traverse(val)

    traverse(dictionary)
    return keys


def validate_json_structure(
    unique_pie_info,
    required_field
):
    for i, pie_info in enumerate(unique_pie_info):
        list_of_keys = list(extract_keys(pie_info['structure']))
        print(type(list_of_keys))
        for field in required_field:
            if field not in list_of_keys:
                unique_pie_info[i]['valid'] = False
    return unique_pie_info


def validate_and_transform_bfs(
    json_dict: dict,
    invalid_values: Dict[str, list[str]],
    transformations: Dict[str, Callable],
    inplace: bool = False,
    **kwargs
) -> tuple[bool, dict]:
    """
    Validates and transforms a JSON dictionary using BFS traversal.

    Args:
        json_dict (dict): Input JSON dictionary
        invalid_values (dict): Dictionary mapping field names to lists of invalid values
        transformations (dict): Dictionary mapping field names to transformation functions
        inplace (bool): Whether to modify the input dictionary or create a copy

    Returns:
        tuple[bool, dict]: A tuple containing:
            - bool: Whether the JSON is valid (True) or contains invalid values (False)
            - dict: The transformed JSON dictionary
    """

    if not inplace:
        json_dict = deepcopy(json_dict)

    is_valid = True
    queue = deque([(None, None, json_dict)])

    while queue:
        _, _, current_dict = queue.popleft()
        if not isinstance(current_dict, dict):
            continue
        for key, value in current_dict.items():
            if key in invalid_values:
                if isinstance(value, list):
                    if any(val in invalid_values[key] for val in value):
                        is_valid = False
                else:
                    if value in invalid_values[key]:
                        is_valid = False

            if key in transformations and is_valid:
                if isinstance(value, list):
                    transformed_values = []
                    for val in value:
                        try:
                            specific_kwargs = kwargs.get(key, {})

                            transformed_val = transformations[key](
                                val, ** specific_kwargs)
                            transformed_values.append(transformed_val)
                        except Exception as e:
                            raise e
                            # transformed_values.append(val)
                    current_dict[key] = transformed_values
                else:
                    try:
                        specific_kwargs = kwargs.get(key, {})
                        current_dict[key] = transformations[key](
                            value, **specific_kwargs)
                    except Exception as e:
                        raise e

            if isinstance(value, dict):
                queue.append((current_dict, key, value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        queue.append((current_dict, key, item))
    return is_valid, json_dict


def find_value_based_field(dictionary, field):
    queue = deque([dictionary])
    while queue:
        current_dict = queue.popleft()

        for key, value in current_dict.items():
            if key == field:
                return value

            if isinstance(value, dict):
                queue.append(value)

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        queue.append(item)
    return None


def image_to_list(image: np.ndarray) -> List:
    """Convert numpy image to list format for JSON serialization"""
    if image is None:
        return None
    return image.tolist()


def list_to_image(image_list: List) -> np.ndarray:
    """Convert list back to numpy image"""
    if image_list is None:
        return None
    return np.array(image_list, dtype=np.uint8)


def main():
    import os
    import sys
    config = Config()

    annotation_path = './ChartQA/ChartQA/ChartQA/ChartQA Dataset/ChartQA Dataset/train/annotations'
    images_path = './ChartQA/ChartQA/ChartQA/ChartQA Dataset/ChartQA Dataset/train/png'
    debug__path = './ChartQA/ChartQA/ChartQA/ChartQA Dataset/ChartQA Dataset/debug'

    save_json_annotation = './ChartQA/ChartQA/ChartQA/ChartQA Dataset/ChartQA Dataset/debug/json_annotation_list.json'
    os.makedirs(debug__path, exist_ok=True)

    # Reading all the annotations
    json_annotation_list: List[Dict[str, Union[str, dict]]] = []

    if os.path.exists(save_json_annotation):
        print("Loading existing JSON annotation list...")
        with open(save_json_annotation, 'r') as f:
            loaded_data = json.load(f)
            json_annotation_list = [
                {
                    'json_name': item['json_name'],
                    'full_path': item['full_path'],
                    'data': item['data'],
                    'image_path': item['image_path']
                }
                for item in loaded_data
            ]
    else:
        print("Creating new JSON annotation list...")
        for filename in tqdm(os.listdir(annotation_path), desc='Reading image and annotation...'):
            full_path = os.path.join(annotation_path, filename)
            with open(full_path, 'r') as f:
                json_dict = json.load(f)
            image_path = os.path.join(
                images_path, filename.replace(".json", ".png"))
            information = {
                'json_name': filename,
                'full_path': full_path,
                'data': json_dict,
                'image_path': image_path
            }
            json_annotation_list.append(information)

        print("Saving JSON annotation list...")

        # with open(save_json_annotation, 'w') as f:
        #     json.dump(json_annotation_list, f, indent=4)

    print(f"Total amount of annotations: {len(json_annotation_list)}")
    print(f"Pie Annotation only program")
    type_pie_annot = [
        info for info in json_annotation_list if info['data']['type'] == 'pie']

    print(f"The number of pie annotation: {len(type_pie_annot)}")

    data_pie_only = [info['data'] for info in type_pie_annot]

    len_unique_struct, unique_pie = find_unique_structures(
        data_pie_only,
        type_pie_annot
    )

    print(f"Number of unique structure found: {len_unique_struct}")

    for i, (key, value) in enumerate(unique_pie.items()):
        pprint(f"{i}.  with {len(value)} in length of list")
        pprint(key)
        print()

    unique_pie_info = []
    for pie_i in unique_pie.keys():
        unique_pie_info.append(
            {
                'structure': json.loads(pie_i),
                'value_json_list': {
                    'data': unique_pie[pie_i],
                    'valid': [True] * len(unique_pie[pie_i])
                },
                'valid': True
            }
        )

    # TODO: FIrst step, check for field validation, so the structure must contain these fields, in order to be considered as valid

    print(
        f"1. Check for the structure if it has all the valid fields: {config.required_field}")

    unique_pie_info_validate = validate_json_structure(
        unique_pie_info,
        config.required_field
    )

    for i, pie_info in enumerate(unique_pie_info_validate):
        print(f"{i+1}, structure validation: {pie_info['valid']} ")
        if not pie_info['valid']:
            print(f"The structure {i+1} will be discarded")

    # TODO: Ok, now check for the invalid value of each field in the valid structure

    for i, pie_info in enumerate(unique_pie_info_validate):
        if not pie_info['valid']:
            continue

        val_json_l = pie_info['value_json_list']
        datas = val_json_l['data']
        valid = val_json_l['valid']

        for j, data in enumerate(datas):
            json_value = data['data']
            image_path = data['image_path']
            image = cv2.imread(image_path)
            image_size = image.shape[:2]
            isValid, transformed_json = validate_and_transform_bfs(
                json_value,
                config.not_valid_value,
                config.transformation,
                bbox={'original_size': image_size,
                      'desired_size': config.resize_image_size},
                text_bbox={'original_size': image_size,
                           'desired_size': config.resize_image_size}
            )

            valid[j] = isValid
            data['data'] = transformed_json

        unique_pie_info_validate[i]['value_json_list']['valid'] = valid

    unique_pie_info_validate_full = []
    for pie_info in unique_pie_info_validate:
        if not pie_info['valid']:
            continue

        unique_pie_info_validate_full.append(
            {
                'structure': pie_info['structure'],
                'value_json_list': [data for data, valid in zip(
                    pie_info['value_json_list']['data'],
                    pie_info['value_json_list']['valid']
                ) if valid]
            }
        )

    print(IMAGE_ERROR)

    for i, pie_info in enumerate(unique_pie_info_validate_full):
        print("=" * 50)
        structure = pie_info['structure']
        print(f"The {i+1} structure: ")
        pprint(structure)
        print(f"List of unique fields: {extract_keys(structure)}")

        datas = pie_info['value_json_list']
        print(f"Length json dict: {len(datas)}")
        print("\nUnique Color Transformations:")
        print("-" * 50)
        all_transformations = set()

        for pie_info in unique_pie_info_validate_full:

            for data in pie_info['value_json_list']:
                transformations = print_transformed_values(
                    data['data'], 'color')

                image_path = data['image_path']
                image_filename = os.path.basename(image_path)
                image = cv2.imread(image_path)
                image = cv2.resize(image, config.resize_image_size)

                annot_image = visualize_image(image, data['data'])
                debug_image_path = os.path.join(debug__path, 'images')
                os.makedirs(debug_image_path, exist_ok=True)
                image_save_path = os.path.join(
                    debug_image_path, image_filename)
                cv2.imwrite(image_save_path, annot_image)

                all_transformations.update(transformations)

        print(f"Found {len(all_transformations)} unique color transformations:")
        print(all_transformations)


if __name__ == "__main__":
    main()
