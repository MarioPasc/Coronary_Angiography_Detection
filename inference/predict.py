#!/usr/bin/env python3
# 8/11/2024.
# Make inference on a given image using a *.pt YOLO model.
# Example of usage:
#      python predict.py ./Inference_try_images/p11_v5_00033.png --output_dir ./Inference_try_images --imgsz 640
# We recommend using an imgsz value of 640, since the model has been trained for this image resolution, regardless of the size of the input image
# YOLO head has been coded to work with different image sizes, but the model performs good with this resolution.
# The available parameters are included in the global variable DEFAULT_PARAMS.
# If you encounter any issues, please contact pascualgonzalez.mario@uma.es

import sys
import subprocess
import importlib
import logging
from typing import Any, Dict, Union, Tuple

# Set up logging
logging.basicConfig(filename='prediction_log.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def install_and_import(package_name: str, import_name: str = None) -> None:
    """
    Checks if a package is installed; if not, installs it.

    Args:
        package_name (str): The package name to install via pip.
        import_name (str, optional): The name used to import the package. Defaults to None.
    """
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        logging.debug(f"Module {import_name} is already installed.")
    except ImportError:
        logging.info(f"Module {import_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logging.info(f"Module {import_name} installed.")
    finally:
        globals()[import_name] = importlib.import_module(import_name)


# Check and install necessary modules
install_and_import('ultralytics')
install_and_import('opencv-python-headless', 'cv2')

import argparse
import os

# Global default parameters for the predict method and model path
DEFAULT_PARAMS: Dict[str, Any] = {
    'conf': 0.25,
    'iou': 0.7,
    'imgsz': 640,
    'half': False,
    'device': None,
    'max_det': 300,
    'vid_stride': 1,
    'stream_buffer': False,
    'visualize': False,
    'augment': False,
    'agnostic_nms': False,
    'classes': None,
    'retina_masks': False,
    'embed': None,
    'project': None,
    'name': None,
    'show': False,
    'save': True,
    'save_frames': False,
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True,
    'line_width': None,
}

VISUALIZATION_PARAMS: Dict[str, Any] = {
    'conf': True,
    'line_width': 1,
    'font_size': 12,
    'font': 'Helvetica.ttf',
    'pil': False,
    'img': None,
    'im_gpu': None,
    'kpt_radius': 3,
    'kpt_line': True,
    'labels': True,
    'boxes': True,
    'probs': False,
    'show': False,
    'save': True,
    'filename': 'annotated_image.png',  
    'color_mode': 'class',
}

MODEL_PATH: str = '../models/iteration2.pt'  # Replace with your actual model path


def parse_imgsz(value: str) -> Union[int, Tuple[int, int]]:
    """
    Parses the image size argument.

    Args:
        value (str): The image size value as a string.

    Returns:
        Union[int, Tuple[int, int]]: The image size as an int or tuple.
    """
    if ',' in value:
        return tuple(map(int, value.split(',')))
    else:
        return int(value)


def main() -> None:
    """
    Main function to run the YOLO prediction with custom parameters.
    """
    try:
        parser = argparse.ArgumentParser(description='Run Ultralytics YOLO prediction with custom parameters.')

        # Positional argument for the image path
        parser.add_argument('source', help='Path to the input image.')

        # Optional argument for the output directory
        parser.add_argument('--output_dir', default='runs/predict', help='Directory to save the output images.')

        # Inference arguments
        parser.add_argument('--conf', type=float, default=DEFAULT_PARAMS['conf'],
                            help='Sets the minimum confidence threshold for detections.')
        parser.add_argument('--iou', type=float, default=DEFAULT_PARAMS['iou'],
                            help='Intersection Over Union (IoU) threshold for NMS.')
        parser.add_argument('--imgsz', type=parse_imgsz, default=DEFAULT_PARAMS['imgsz'],
                            help='Image size for inference. Can be single int or height,width tuple.')
        parser.add_argument('--half', action='store_true', help='Enables half-precision (FP16) inference.')
        parser.add_argument('--device', default=DEFAULT_PARAMS['device'],
                            help='Device for inference (e.g., cpu, cuda:0).')
        parser.add_argument('--max_det', type=int, default=DEFAULT_PARAMS['max_det'],
                            help='Maximum number of detections per image.')
        parser.add_argument('--vid_stride', type=int, default=DEFAULT_PARAMS['vid_stride'],
                            help='Frame stride for video inputs.')
        parser.add_argument('--stream_buffer', action='store_true',
                            help='Queue incoming frames for video streams.')
        parser.add_argument('--visualize', action='store_true',
                            help='Activates visualization of model features.')
        parser.add_argument('--augment', action='store_true', help='Enables test-time augmentation (TTA).')
        parser.add_argument('--agnostic_nms', action='store_true', help='Enables class-agnostic NMS.')
        parser.add_argument('--classes', nargs='+', type=int, default=DEFAULT_PARAMS['classes'],
                            help='Filter predictions to specified class IDs.')
        parser.add_argument('--retina_masks', action='store_true',
                            help='Uses high-resolution segmentation masks.')
        parser.add_argument('--embed', nargs='+', type=int, default=DEFAULT_PARAMS['embed'],
                            help='Layers to extract feature vectors from.')
        parser.add_argument('--project', default=DEFAULT_PARAMS['project'],
                            help='Project directory for saving outputs.')
        parser.add_argument('--name', default=DEFAULT_PARAMS['name'],
                            help='Name of the prediction run.')

        # Visualization arguments
        parser.add_argument('--show', action='store_true', help='Display annotated images or videos.')
        parser.add_argument('--nosave', action='store_false', dest='save',
                            help='Do not save annotated images or videos.')
        parser.add_argument('--save_frames', action='store_true',
                            help='Save individual frames when processing videos.')
        parser.add_argument('--save_txt', action='store_true', help='Save detection results in a text file.')
        parser.add_argument('--save_conf', action='store_true',
                            help='Include confidence scores in saved text files.')
        parser.add_argument('--save_crop', action='store_true', help='Save cropped images of detections.')
        parser.add_argument('--no_show_labels', action='store_false', dest='show_labels',
                            help='Do not display labels.')
        parser.add_argument('--no_show_conf', action='store_false', dest='show_conf',
                            help='Do not display confidence scores.')
        parser.add_argument('--no_show_boxes', action='store_false', dest='show_boxes',
                            help='Do not draw bounding boxes.')
        parser.add_argument('--line_width', type=int, default=DEFAULT_PARAMS['line_width'],
                            help='Line width of bounding boxes.')

        args = parser.parse_args()
        logging.debug(f"Parsed arguments: {args}")

        # Load the model
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        logging.debug(f"Model loaded from {MODEL_PATH}")

        # Prepare parameters for the predict method
        args_dict: Dict[str, Any] = vars(args)
        output_dir: str = args_dict.pop('output_dir')

        # Remove arguments not accepted by the predict method
        params: Dict[str, Any] = {k: v for k, v in args_dict.items() if k in DEFAULT_PARAMS}

        # Add 'source' to params
        params['source'] = args.source

        params['save'] = False  # We handle saving manually

        # Adjust 'device' parameter
        if params['device'] == 'None' or params['device'] == 'cpu':
            params['device'] = 'cpu'
        elif params['device'] is None:
            params['device'] = None

        logging.debug(f"Prediction parameters: {params}")

        # Prediction
        results = model.predict(**params)
        logging.debug("Prediction completed.")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logging.debug(f"Output directory ensured at {output_dir}")

        # Process and save the results
        import cv2
        for result in results:
            image_path: Union[str, os.PathLike] = result.path
            image_base_name: str = os.path.splitext(os.path.basename(image_path))[0]
            output_filename: str = f'predict_{image_base_name}.png'
            output_path: Union[str, os.PathLike] = os.path.join(output_dir, output_filename)
            VISUALIZATION_PARAMS['filename'] = output_path
            annotated_image = result.plot(**VISUALIZATION_PARAMS)
            cv2.imwrite(output_path, annotated_image)
            logging.debug(f"Result saved to {output_path}")

    except Exception as e:
        logging.exception("An error occurred during prediction.")
        print(f"An error occurred: {e}. Please check the log file for more details.")


if __name__ == "__main__":
    main()
