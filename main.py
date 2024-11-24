
from lib import *
from utils import *
from transformations import *
from config import *

def main():

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
