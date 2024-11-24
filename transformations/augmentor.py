
from lib import *
from utils.path import DataPaths, PlotBoundingBox

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



