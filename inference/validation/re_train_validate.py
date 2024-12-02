import os
from typing import Union, Dict, List
import logging
import shutil

# Configure logging
logging.basicConfig(filename='prediction_log.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

LABEL_DICT: Dict[str, int] = {
    "p0_20": 0, "p20_50": 0, "p50_70": 0,
    "p70_90": 0, "p90_98": 0, "p99": 0, "p100": 0
}


class Formatting:
    def __init__(self, 
                 videos_folder: Union[str, os.PathLike],
                 dataset_folder: Union[str, os.PathLike] = "./validation_yolo_format") -> None:
        """
        Create a formatting instance to convert videos to YOLO format.
        
        Args:
        ----------------
            videos_folder (Union[str, os.PathLike]): Path to the folder containing videos of the patients.
            dataset_folder (Union[str, os.PathLike]): Path to the YOLO dataset output folder.
        """
        self.videos_folder = videos_folder
        self.dataset_folder = dataset_folder

    def _create_directory_structure(self, output_dir: str) -> None:
        """
        Create the YOLO dataset directory structure for validation.

        Args:
        -------
        output_dir : str
            Path to the output directory for the dataset.
        """
        try:
            os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)
            logging.info("Directory structure created successfully.")
        except Exception as e:
            logging.error(f"Error creating directory structure: {e}")
            raise

    def _read_selected_frames(self, selected_frames_file: str) -> List[str]:
        """
        Read the selected frames from the file.

        Args:
        -------
        selected_frames_file : str
            Path to the *_selectedFrames.txt file.

        Returns:
        -------
        List[str]
            List of selected frame names without extensions.
        """
        try:
            with open(selected_frames_file, "r") as f:
                selected_frames = [line.strip() for line in f.readlines()]
            logging.info(f"Selected frames read from {selected_frames_file}.")
            return selected_frames
        except Exception as e:
            logging.error(f"Error reading selected frames: {e}")
            raise

    def _copy_images(self, input_path: str, output_path: str, selected_frames: List[str]) -> None:
        """
        Copy images from the input directory to the YOLO dataset structure, filtering by selected frames.

        Args:
        -------
        input_path : str
            Path to the input directory containing images.
        output_path : str
            Path to the output directory for images.
        selected_frames : List[str]
            List of selected frame names without extensions.
        """
        try:
            for img_file in os.listdir(input_path):
                if img_file.endswith(".png") and os.path.splitext(img_file)[0] in selected_frames:
                    shutil.copy(os.path.join(input_path, img_file), os.path.join(output_path, img_file))
            logging.info(f"Filtered images copied from {input_path} to {output_path}.")
        except Exception as e:
            logging.error(f"Error copying images: {e}")
            raise

    @staticmethod
    def _convertBboxFormatToYOLO(
        bbox: str, 
        img_width: int, 
        img_height: int, 
        class_mappings: Dict[str, int] = LABEL_DICT
    ) -> str:
        """
        Convert bounding box from [x, y, w, h, class] format to YOLO format.

        Args
        -------------
        bbox : str
            Bounding box in the format "x y w h class".
        img_width : int
            Width of the image.
        img_height : int
            Height of the image.

        Returns
        -------------
        str
            Bounding box in YOLO format "class x_center y_center width height".
        """
        x, y, w, h, cls = bbox.split()
        x, y, w, h = int(x), int(y), int(w), int(h)
        cls = class_mappings[cls]
        
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w /= img_width
        h /= img_height
        
        return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

    def _convert_groundtruths_to_yolo(
        self,
        groundtruth_path: str, 
        labels_output_path: str, 
        img_width: int, 
        img_height: int,
        selected_frames: List[str]
    ) -> None:
        """
        Convert groundtruth files to YOLO format and save them in the labels directory, filtering by selected frames.

        Args:
        -------
        groundtruth_path : str
            Path to the directory containing groundtruth files.
        labels_output_path : str
            Path to the output directory for YOLO labels.
        img_width : int
            Width of the images.
        img_height : int
            Height of the images.
        selected_frames : List[str]
            List of selected frame names without extensions.
        """
        try:
            for gt_file in os.listdir(groundtruth_path):
                frame_name = os.path.splitext(gt_file)[0]
                if gt_file.endswith(".txt") and frame_name in selected_frames:
                    input_file_path = os.path.join(groundtruth_path, gt_file)
                    output_file_path = os.path.join(labels_output_path, gt_file)
                    
                    with open(input_file_path, "r") as f:
                        lines = f.readlines()
                    
                    yolo_lines = [
                        self._convertBboxFormatToYOLO(line.strip(), img_width, img_height)
                        for line in lines
                    ]
                    
                    with open(output_file_path, "w") as f:
                        f.writelines(yolo_lines)
                    
                    logging.info(f"Converted groundtruths in {gt_file} to YOLO format.")
        except Exception as e:
            logging.error(f"Error converting groundtruths: {e}")
            raise

    def _generate_config_yaml(self, train_path: str = "", test_path: str = "") -> None:
        """
        Generate a YAML configuration file for YOLO.

        Args:
        -------
        train_path : str
            Path to the training images folder (default is empty since no training data is provided).
        test_path : str
            Path to the testing images folder (default is empty since no testing data is provided).
        """
        # Generate names in YOLO format
        names = {idx: class_name for class_name, idx in LABEL_DICT.items()}
        
        if len(names) == 1:
            names = {0: 'lesion'}

        config_file = os.path.join(self.dataset_folder, "..", "data.yaml")
        try:
            with open(config_file, "w") as f:
                # Write YAML content manually to include a blank line
                f.write(f"path: {self.dataset_folder}\n")
                f.write(f"train: {train_path or 'images/train'}\n")
                f.write(f"val: images/val\n")
                f.write(f"test: {test_path or 'images/test'}\n\n")  # Add blank line here
                f.write("names:\n")
                for idx, name in enumerate(names.values()):
                    f.write(f"  {name}: {idx}\n")
            logging.info(f"Generated config.yaml at {config_file}.")
        except Exception as e:
            logging.error(f"Error generating config.yaml: {e}")
            raise


    def prepare_yolo_validation_dataset(
        self,
        img_width: int, 
        img_height: int
    ) -> None:
        """
        Prepare YOLO validation dataset from the given input directory structure.

        Args:
        -------
        img_width : int
            Width of the images.
        img_height : int
            Height of the images.
        """
        try:
            self._create_directory_structure(self.dataset_folder)
            images_output_path = os.path.join(self.dataset_folder, "images/val")
            labels_output_path = os.path.join(self.dataset_folder, "labels/val")
            
            video_dirs = [
                d for d in os.listdir(self.videos_folder) 
                if os.path.isdir(os.path.join(self.videos_folder, d))
            ]
            
            for video_dir in video_dirs:
                video_path = os.path.join(self.videos_folder, video_dir)
                input_path = os.path.join(video_path, "input")
                groundtruth_path = os.path.join(video_path, "groundtruth")
                selected_frames_file = [os.path.join(video_path, file) for file in os.listdir(video_path) if file.endswith('_selectedFrames.txt')][0]

                if not os.path.isfile(selected_frames_file):
                    logging.warning(f"Selected frames file not found for {video_dir}, skipping.")
                    continue
                
                selected_frames = self._read_selected_frames(selected_frames_file)
                self._copy_images(input_path, images_output_path, selected_frames)
                self._convert_groundtruths_to_yolo(
                    groundtruth_path, labels_output_path, img_width, img_height, selected_frames
                )
                self._generate_config_yaml()
            logging.info("YOLO validation dataset prepared successfully.")
        except Exception as e:
            logging.error(f"Error preparing YOLO validation dataset: {e}")
            raise

formatter = Formatting(videos_folder="./videos")
formatter.prepare_yolo_validation_dataset(img_width=512, img_height=512)
