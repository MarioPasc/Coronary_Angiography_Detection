import os
from typing import Union, Dict, List, Tuple
import logging
import shutil
import yaml
import json
from ultralytics import YOLO  

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
        Create the YOLO dataset directory structure for validation and training.

        Args:
        -------
        output_dir : str
            Path to the output directory for the dataset.
        """
        try:
            for split in ['train', 'val']:
                os.makedirs(os.path.join(output_dir, f"images/{split}"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f"labels/{split}"), exist_ok=True)
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

    def _copy_images(self, input_path: str, output_paths: List[str], selected_frames: List[str]) -> None:
        """
        Copy images from the input directory to the YOLO dataset structure, filtering by selected frames.

        Args:
        -------
        input_path : str
            Path to the input directory containing images.
        output_paths : List[str]
            List of paths to the output directories for images.
        selected_frames : List[str]
            List of selected frame names without extensions.
        """
        try:
            for img_file in os.listdir(input_path):
                if img_file.endswith(".png") and os.path.splitext(img_file)[0] in selected_frames:
                    for output_path in output_paths:
                        shutil.copy(os.path.join(input_path, img_file), os.path.join(output_path, img_file))
            logging.info(f"Filtered images copied from {input_path} to {', '.join(output_paths)}.")
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
        labels_output_paths: List[str], 
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
        labels_output_paths : List[str]
            List of paths to the output directories for YOLO labels.
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
                    
                    with open(input_file_path, "r") as f:
                        lines = f.readlines()
                    
                    yolo_lines = [
                        self._convertBboxFormatToYOLO(line.strip(), img_width, img_height)
                        for line in lines
                    ]
                    
                    for labels_output_path in labels_output_paths:
                        output_file_path = os.path.join(labels_output_path, gt_file)
                        with open(output_file_path, "w") as f:
                            f.writelines('\n'.join(yolo_lines))
                    
                    logging.info(f"Converted groundtruths in {gt_file} to YOLO format and saved to {', '.join(labels_output_paths)}.")
        except Exception as e:
            logging.error(f"Error converting groundtruths: {e}")
            raise

    def _generate_config_yaml(self) -> None:
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
                f.write(f"train: {'images/train'}\n")
                f.write(f"val: images/val\n")
                f.write(f"test: {'images/test'}\n\n")  
                f.write("names:\n")
                for idx, name in enumerate(names.values()):
                    f.write(f"  {idx}: {name}\n")
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
        Prepare YOLO validation and training dataset from the given input directory structure.

        Args:
        -------
        img_width : int
            Width of the images.
        img_height : int
            Height of the images.
        """
        try:
            self._create_directory_structure(self.dataset_folder)
            images_output_paths = [
                os.path.join(self.dataset_folder, "images/train"),
                os.path.join(self.dataset_folder, "images/val")
            ]
            labels_output_paths = [
                os.path.join(self.dataset_folder, "labels/train"),
                os.path.join(self.dataset_folder, "labels/val")
            ]
            
            video_dirs = [
                d for d in os.listdir(self.videos_folder) 
                if os.path.isdir(os.path.join(self.videos_folder, d))
            ]
            
            for video_dir in video_dirs:
                video_path = os.path.join(self.videos_folder, video_dir)
                input_path = os.path.join(video_path, "input")
                groundtruth_path = os.path.join(video_path, "groundtruth")
                selected_frames_files = [os.path.join(video_path, file) for file in os.listdir(video_path) if file.endswith('_selectedFrames.txt')]
                
                if not selected_frames_files:
                    logging.warning(f"Selected frames file not found for {video_dir}, skipping.")
                    continue

                selected_frames_file = selected_frames_files[0]
                selected_frames = self._read_selected_frames(selected_frames_file)
                self._copy_images(input_path, images_output_paths, selected_frames)
                self._convert_groundtruths_to_yolo(
                    groundtruth_path, labels_output_paths, img_width, img_height, selected_frames
                )
            self._generate_config_yaml()
            logging.info("YOLO validation and training dataset prepared successfully.")
        except Exception as e:
            logging.error(f"Error preparing YOLO dataset: {e}")
            raise

class YOLOTrainer:
    def __init__(self, model_path: str, data_yaml: str, args_yaml: str):
        """
        Initialize the YOLOTrainer with a given model and data configuration.

        Args:
        -------
        model_path : str
            Path to the YOLO model (.pt file).
        data_yaml : str
            Path to the data.yaml file.
        args_yaml : str
            Path to the args.yaml file containing hyperparameters.
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.args_yaml = args_yaml
        self.model = YOLO(model_path)
        self.hyperparameters = self._load_hyperparameters()

    def _load_hyperparameters(self):
        """
        Load hyperparameters from the args.yaml file, excluding certain keys.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of hyperparameters to use for training.
        """
        try:
            with open(self.args_yaml, 'r') as f:
                args = yaml.safe_load(f)
            logging.info(f"Loaded hyperparameters from {self.args_yaml}")
        except Exception as e:
            logging.error(f"Error loading hyperparameters: {e}")
            raise

        # Parameters to exclude
        exclude_keys = {'task', 'mode', 'model', 'data', 'epochs', 'name', 'save_dir', 'resume', 'weights', 'project'}
        hyperparameters = {k: v for k, v in args.items() if k not in exclude_keys}
        return hyperparameters

    def validate_model(self, imgsz: int = 512, batch: int = 16, iou: float = 0.6, save_json: bool = True, save_dir: str = 'runs/val'):
        """
        Validate the model using the validation dataset and save metrics to a JSON file.

        Args:
        -------
        imgsz : int
            Image size for validation.
        batch : int
            Batch size for validation.
        iou : float
            IoU threshold for validation.
        save_json : bool
            Whether to save detection results in COCO JSON format.
        save_dir : str
            Directory to save validation results.
        """
        # Perform validation
        validator = self.model.val(
            data=self.data_yaml,
            imgsz=imgsz,
            batch=batch,
            iou=iou,
            save_json=save_json,
            save_dir=save_dir
        )

        # Access metrics directly from the validator object
        metrics_dict = {
            'precision': float(validator.box.mp),            # Mean Precision over all classes
            'recall': float(validator.box.mr),               # Mean Recall over all classes
            'map50': float(validator.box.map50),             # Mean AP@0.5 over all classes
            'map': float(validator.box.map),                 # Mean AP@0.5:0.95 over all classes
            'map75': float(validator.box.map75),             # Mean AP@0.75 over all classes
            'per_class_precision': validator.box.p.tolist(),   # List of precision per class
            'per_class_recall': validator.box.r.tolist(),      # List of recall per class
            'maps': validator.box.maps.tolist(),             # mAPs for each class (converted to list)
        }

        # Save metrics to JSON file
        metrics_file = os.path.join(save_dir, 'validation_metrics.json')
        os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        logging.info(f"Validation metrics saved to {metrics_file}")

    def train_model(self, epochs: int = 10, imgsz: int = 512, batch: int = 16, name: str = 'retrain_CADICA', save_dir: str = 'runs/train'):
        """
        Continue training the model using the training dataset.

        Args:
        -------
        epochs : int
            Number of epochs to train.
        imgsz : int
            Image size for training.
        batch : int
            Batch size for training.
        name : str
            Name of the training run.
        save_dir : str
            Directory to save training results.
        """
        # Prepare training arguments
        train_args = {
            'model': self.model_path,
            'data': self.data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'name': name,
            'save_dir': save_dir,
            'resume': False,  
            'val': False
        }

        # Merge with hyperparameters, hyperparameters have lower priority
        train_args.update(self.hyperparameters)

        # Log the training arguments
        logging.info(f"Training with arguments: {train_args}")

        # Start training
        self.model.train(**train_args)
def main() -> None:
    
    CADICA_image_size: Tuple[int, int] = (512, 512)
    epochs: int = 10
    batch_size: int = 16
    videos_folder_path: Union[str, os.PathLike] = "./videos"
    model_path: Union[str, os.PathLike] = './iteration_2.pt'  
    model_args: Union[str, os.PathLike] = './args.yaml'
    
    formatter = Formatting(videos_folder=videos_folder_path)
    formatter.prepare_yolo_validation_dataset(img_width=CADICA_image_size[0], img_height=CADICA_image_size[1])

    data_yaml: os.PathLike = os.path.join(formatter.dataset_folder, "..", "data.yaml")

    model = YOLOTrainer(model_path=model_path, data_yaml=data_yaml, args_yaml=model_args)
    model.validate_model(imgsz=max(CADICA_image_size), batch=batch_size)
    model.train_model(epochs=epochs, imgsz=max(CADICA_image_size), batch=batch_size)
    model.validate_model(imgsz=max(CADICA_image_size), batch=batch_size)
    

if __name__ == "__main__":
    main()