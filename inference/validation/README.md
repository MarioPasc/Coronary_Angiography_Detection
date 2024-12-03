# YOLO Model Validation and Retraining Pipeline

This submodule provides a pipeline for:

- Preparing video data into a format suitable for training and validating YOLO models.
- Validating an existing YOLO model on a validation dataset.
- Retraining the YOLO model using new training data.

## Table of Contents

- [Requirements](#requirements)
- [Input Data Format](#input-data-format)
- [Required Files](#required-files)
- [Usage Instructions](#usage-instructions)
- [Results Structure](#results-structure)
- [Additional Notes](#additional-notes)

## Requirements

- **Python**: Version 3.9 or higher.
- **Python Packages**:
  - `ultralytics`
  - `pyyaml`

Install the required packages using:

```bash
pip install ultralytics pyyaml
```

## Input Data Format

The input data should be organized in a specific directory structure and format to be compatible with the YOLO training pipeline.

### Directory Structure

- **Videos Folder**: A main directory containing subdirectories for each patient/video. This is the CADICA dataset structure.

```
videos/
├── video1/
│   ├── input/
│   ├── groundtruth/
│   └── video1_selectedFrames.txt
├── video2/
│   ├── input/
│   ├── groundtruth/
│   └── video2_selectedFrames.txt
└── ...
```

#### Inside Each Video Subdirectory

1. **`input/`**: Contains image frames extracted from the video in PNG format.
   - Example:
     ```
     input/
     ├── frame1.png
     ├── frame2.png
     ├── frame3.png
     └── ...
     ```

2. **`groundtruth/`**: Contains ground truth annotation files corresponding to each image frame.
   - Each annotation file is a text file with the same name as the image file (but with a `.txt` extension).
   - Example:
     ```
     groundtruth/
     ├── frame1.txt
     ├── frame2.txt
     ├── frame3.txt
     └── ...
     ```

3. **`*_selectedFrames.txt`**: A text file listing the selected frame filenames (without extension) to be included in the dataset.
   - The filename should end with `_selectedFrames.txt` (e.g., `video1_selectedFrames.txt`).
   - Contents example:
     ```
     frame1
     frame3
     frame5
     ```

### Ground Truth Annotation Format

Each `.txt` file in the `groundtruth/` directory should contain bounding box annotations in the following format:

```
x y w h class_label
```

- **`x`**: X-coordinate of the top-left corner of the bounding box.
- **`y`**: Y-coordinate of the top-left corner of the bounding box.
- **`w`**: Width of the bounding box.
- **`h`**: Height of the bounding box.
- **`class_label`**: Class label as a string (e.g., "p0_20", "p20_50", etc.).

Example content of `frame1.txt`:

```
100 150 50 75 p0_20
200 250 60 80 p50_70
```

### Class Labels

The class labels used in the ground truth files should match the keys in the `LABEL_DICT` defined in the code:

```python
LABEL_DICT = {
    "p0_20": 0,
    "p20_50": 0,
    "p50_70": 0,
    "p70_90": 0,
    "p90_98": 0,
    "p99": 0,
    "p100": 0
}
```

**Note**: All class labels are mapped to the same class index `0` in this case. This is because the model given is expected to be a detection model.

## Required Files

To use the code effectively, you need the following files:

1. **Input Data**: Organized as described in the [Input Data Format](#input-data-format) section.

2. **Pre-trained YOLO Model**: A `.pt` file containing the pre-trained YOLO model you wish to validate and retrain (e.g., `iteration_2.pt`).

3. **Hyperparameters File (`args.yaml`)**: A YAML file containing hyperparameters for training. Exclude parameters like `epochs`, `data`, `model`, `name`, `save_dir`, etc., as these are set within the code.

   Example `args.yaml`:

   ```yaml
   optimizer: Adam
   lr0: 0.001
   momentum: 0.9
   weight_decay: 0.0005
   batch: 16
   imgsz: 512
   # ... other hyperparameters
   ```

## Usage Instructions

Follow these steps to run the pipeline:

### 1. Prepare the Input Data

- Ensure your input data is organized according to the [Input Data Format](#input-data-format).
- Place your data in a directory (default is `./videos/`).

### 2. Update the Code Configuration

- Open the script file (e.g., `re_train_validate.py`).
- Modify the `main()` function parameters if necessary:

  ```python
  def main() -> None:
      CADICA_image_size: Tuple[int, int] = (512, 512)  # Update if your images have a different size
      epochs: int = 10  # Number of training epochs
      batch_size: int = 16  # Batch size for training and validation
      videos_folder_path: Union[str, os.PathLike] = "./videos"  # Path to your videos folder
      model_path: Union[str, os.PathLike] = './iteration_2.pt'  # Path to your pre-trained model
      model_args: Union[str, os.PathLike] = './args.yaml'  # Path to your hyperparameters file
      # ... rest of the code
  ```

### 3. Run the Script

- Execute the script using the command line:

  ```bash
  python re_train_validate.py
  ```

### 4. Process Overview

The script performs the following steps:

1. **Data Formatting**:
   - Converts the input data into YOLO-compatible format.
   - Creates the necessary directory structure.
   - Generates the `data.yaml` configuration file.

2. **Model Validation**:
   - Validates the existing model using the validation dataset.
   - Saves validation metrics to `validation_metrics.json`.

3. **Model Training**:
   - Retrains the model using the training dataset and specified hyperparameters.

4. **Post-Training Validation**:
   - Validates the retrained model.
   - Saves updated validation metrics.

## Results Structure

After running the script, the following directories and files will be generated:

### 1. YOLO Dataset Directory

- Located at `./validation_yolo_format/`.
- Contains the formatted dataset ready for YOLO training and validation.

```
validation_yolo_format/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

### 2. Configuration File

- `data.yaml`: Located in the parent directory of `validation_yolo_format/`.
- Specifies paths to training and validation data and class names.

Example `data.yaml` content:

```yaml
path: ./validation_yolo_format
train: images/train
val: images/val
test: images/test

names:
  0: lesion
```

### 3. Validation Results

- Validation metrics are saved in JSON format.
- Before retraining: `runs/val/validation_metrics.json`
- After retraining: `runs/val2/validation_metrics.json`

Example `validation_metrics.json`:

```json
{
    "precision": 0.85,
    "recall": 0.80,
    "map50": 0.82,
    "map": 0.75,
    "map75": 0.78,
    "per_class_precision": [0.88],
    "per_class_recall": [0.81],
    "maps": [0.75]
}
```

### 4. Training Results

- Training outputs are saved in the `runs/train/` directory under the specified `name` (default is `retrain_CADICA`).

```
runs/train/retrain_CADICA/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── train_batch0.jpg
└── ... other training artifacts
```

## Additional Notes

- **Image Size**: The default image size is set to `(512, 512)`. Adjust `CADICA_image_size` in the `main()` function if your images have different dimensions.

- **Class Mappings**: Ensure that the class labels in your ground truth annotations match those in `LABEL_DICT`. All classes are currently mapped to a single class index `0`.

- **Hyperparameters**: Customize the `args.yaml` file with your desired training hyperparameters. Avoid including parameters that are already specified in the code (e.g., `epochs`, `data`, `model`, `name`, `save_dir`).

- **Logging**: Logs are saved to `prediction_log.log`, providing detailed information about the execution process.

- **Dependencies**: Ensure all required Python packages are installed before running the script.

- **Error Handling**: The script includes error handling and logging to help diagnose issues during execution.

- **Validation IOU Threshold**: The IoU threshold for validation is set to `0.6` by default. Adjust it in the `validate_model` method if needed.

- **Batch Size and Epochs**: Modify `batch_size` and `epochs` in the `main()` function to suit your computational resources and training needs.

## Contact

For any questions or issues, please contact pascualgonzalez.mario@uma.es