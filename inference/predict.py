# 8/11/2024. 
# Make inference on a given image using a *.pt YOLO model. 
# Example of usage:
#      python predict.py ./Inference_try_images/p11_v5_00033.png --output_dir ./Inference_try_images --imgsz 512
# The avaiable parameters are included in the global variable DEFAULT PARAMS.
# If you encounter any issues, please contact pascualgonzalez.mario@uma.es

from ultralytics import YOLO
import argparse
import os
import cv2

# Global default parameters for the predict method and model path
DEFAULT_PARAMS = {
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

MODEL_PATH = './weights/best.pt'  # Replace with your actual model path

def parse_imgsz(value):
    if ',' in value:
        return tuple(map(int, value.split(',')))
    else:
        return int(value)

def main():
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

    # Load the model
    model = YOLO(MODEL_PATH)

    # Prepare parameters for the predict method
    args_dict = vars(args)
    output_dir = args_dict.pop('output_dir')

    # Remove arguments not accepted by the predict method
    params = {k: v for k, v in args_dict.items() if k in DEFAULT_PARAMS}

    # Add 'source' to params
    params['source'] = args.source

    params['save'] = False  # We handle saving manually

    # Adjust 'device' parameter
    if params['device'] == 'None' or params['device'] == 'cpu':
        params['device'] = 'cpu'

    # Prediction
    results = model.predict(**params)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save the results
    for result in results:
        image_path = result.path
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f'predict_{image_base_name}.png'
        output_path = os.path.join(output_dir, output_filename)
        annotated_image = result.plot()
        cv2.imwrite(output_path, annotated_image)

if __name__ == "__main__":
    main()
