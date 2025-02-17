
# ICA Dataset Integration Module

This module is designed to standardize, process, and integrate multiple coronary artery disease (CAD) image datasets into one unified JSON file. The goal is to harmonize the annotations and image metadata from diverse sources—each with its own structure—into a common format that can be easily consumed for training detection models (e.g., using YOLO) or further analysis.

## Overview

The integration module processes the following datasets:
- **CADICA**  
- **ARCADE**  
- **KEMEROVO**

Each dataset is handled by its own submodule that:
- Downloads and extracts the raw data.
- Converts dataset-specific annotations into a unified JSON structure.
- Transforms bounding box coordinates to the YOLO normalized format.
- Constructs unique identifiers for each image/annotation pair that incorporate key metadata (e.g., patient, video, frame, split).

The final JSON produced by this module has a top-level key `"Standard_dataset"` containing entries for every processed image. Each entry includes:

- **id:** A unique identifier in the format:
  - **CADICA:** `cadica_p{patient}_v{video}_{frame}`  
  - **ARCADE:** `arcade_{split}_p{num}_v{num}_{frame}` (with frame padded to 5 digits)  
  - **KEMEROVO:** `kemerovo_p{patient}_v{video}_{frame}` (with frame padded to 5 digits)
- **dataset_origin:** The originating dataset.
- **lesion:** A Boolean flag indicating whether the image contains lesion annotations.
- **image:** A dictionary with:
  - `name`: New image filename.
  - `route`: Full path to the image.
  - `original_name`: Original image filename.
  - `width` and `height`: Image resolution (in pixels).
- **annotations:** A dictionary with:
  - `name`: Annotation file name (ending in `.txt`).
  - Bounding box entries (e.g., `bbox1`, `bbox2`, …) where each bounding box is represented in YOLO normalized coordinates (`x_center`, `y_center`, `width`, `height`) along with its original label.

## Final JSON Format Example

```json
{
  "Standard_dataset": {
    "cadica_p11_v10_00026": {
      "id": "cadica_p11_v10_00026",
      "dataset_origin": "cadica",
      "lesion": true,
      "image": {
        "name": "cadica_p11_v10_00026.png",
        "route": "/path/to/CADICA/CADICA/selectedVideos/p11/v10/input/p11_v10_00026.png",
        "original_name": "p11_v10_00026.png",
        "width": 512,
        "height": 512
      },
      "annotations": {
        "name": "cadica_p11_v10_00026.txt",
        "bbox1": {
          "x_center": 0.456,
          "y_center": 0.532,
          "width": 0.123,
          "height": 0.098,
          "label": "lesion"
        }
      }
    },
    "arcade_train_p98_v98_00098": { ... },
    "kemerovo_p002_v5_00016": { ... }
  }
}
```

## Integration

The integration module (`integrate.py`) provides a function `integrate_datasets()` that accepts:
- A list of dataset names to process (e.g., `["CADICA", "ARCADE", "KEMEROVO"]`).
- A mapping of dataset names to their respective root directories.
- Optional parameters (such as the ARCADE task, which can be `"stenosis"`, `"syntax"`, or `"both"`).

The function calls the dataset-specific processing functions, merges the resulting JSON entries, and outputs a single JSON file containing all standardized entries.

## Dataset Information

| **Dataset** | **Download Link** | **Citation** |
|-------------|-------------------|--------------|
| **CADICA**  | [CADICA](https://data.mendeley.com/datasets/p9bpx9ctcv/2) | Jiménez-Partinen, Ariadna; Molina-Cabello, Miguel A.; Thurnhofer-Hemsi, Karl; Palomo, Esteban; Rodríguez-Capitán, Jorge; Molina-Ramos, Ana I.; Jiménez-Navarro, Manuel (2024), “CADICA: a new dataset for coronary artery disease”, *Mendeley Data*, V2, doi: [10.17632/p9bpx9ctcv.2](https://doi.org/10.17632/p9bpx9ctcv.2) |
| **ARCADE**  | [ARCADE](https://zenodo.org/records/10390295) | Maxim Popov, “ARCADE: Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs Dataset”. Research Institute of Cardiology and Internal Diseases, Dec. 15, 2023. doi: [10.5281/zenodo.10390295](https://doi.org/10.5281/zenodo.10390295) |
| **KEMEROVO**| [KEMEROVO](https://data.mendeley.com/datasets/ydrm75xywg/2) | Danilov, Viacheslav; Klyshnikov, Kirill; Kutikhin, Anton; Gerget, Olga; Frangi, Alejandro; Ovcharenko, Evgeny (2021), “Angiographic dataset for stenosis detection”, *Mendeley Data*, V2, doi: [10.17632/ydrm75xywg.2](https://doi.org/10.17632/ydrm75xywg.2) |

This module is part of the overall **ICA** project for atherosclerosis detection using YOLO. For further details, please refer to the corresponding documentation in each dataset module.

