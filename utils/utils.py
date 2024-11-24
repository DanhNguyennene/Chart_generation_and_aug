
from lib import *



PALETTE = {name: to_rgb(hex)
           for name, hex in matplotlib.colors.TABLEAU_COLORS.items()}

SPECIAL_MAPPING = {
    'lightbrow': "#C4A484",
    'lightbrown': "#C4A484",
    "darkbrown": "#C4A484",
    "brow": "#964B00"
}

IMAGE_ERROR = set()
def generate_random_color_bgr() -> Tuple[int, int, int]:
    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)
    return blue, green, red

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
