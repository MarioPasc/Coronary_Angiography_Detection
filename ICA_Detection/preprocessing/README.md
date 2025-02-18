# Preprocessing Module
---

# planner.py

The **Preprocessing Planner** is responsible for analyzing a standardized dataset (JSON format) and generating a preprocessing plan for each image entry. This plan specifies the steps needed to transform images so that they meet desired criteria before further analysis or model training.

### Overview

The planner inspects each image’s metadata (such as resolution, data type, and file format) and compares them with user-specified target parameters. If an image does not meet the target criteria, the planner adds a corresponding tag in the `"preprocessing_plan"` field of that image's JSON entry.

This unified plan is defined as a JSON (or Python dictionary) with keys representing each preprocessing step and its parameters. For example:

```json
{
  "resolution_standarization": {"desired_X": 1000, "desired_Y": 1000, "method": "bilinear"},
  "dtype_standarization": {"desired_dtype": "uint8"},
  "format_standarization": {"desired_format": "png"},
  "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0}
}
```

When applied, if an image does not have the desired resolution, its entry will be tagged as follows:

```json
"preprocessing_plan": {
  "resolution_standarization": {"desired_X": 1000, "desired_Y": 1000, "method": "bilinear"}
}
```

The plan also unconditionally adds the filtering smoothing equalization step, so every image will have a corresponding tag if that step is provided.

## Functionalities

- **Resolution Standardization:**  
  Checks if the image dimensions match the desired width and height. If not, it adds a tag with the desired dimensions and the interpolation method.

- **Dtype Standardization:**  
  Reads the image file (using OpenCV) and compares its data type with the desired one (e.g., "uint8"). If they differ, a corresponding tag is added.

- **Format Standardization:**  
  Compares the image file format (inferred from the filename) with the desired format. If they do not match, a tag is added specifying the conversion (e.g., from "bmp" to "png").

- **Filtering Smoothing Equalization:**  
  Unconditionally tags each image with the parameters for a filtering smoothing equalization step (e.g., Gaussian window size and sigma). This step will later be applied to enhance image quality.


## Integration

This planner is intended to be used as part of the preprocessing pipeline. After planning, the specific tools in the `preprocessing/tools` folder are applied to the images based on the instructions in the JSON.