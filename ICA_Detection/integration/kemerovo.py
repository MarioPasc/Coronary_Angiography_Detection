# ica_yolo_detection/datasets/dataset_kemerovo.py

import os
import shutil
import requests
import zipfile
import json
from typing import List, Dict, Tuple, Any, Optional
import xml.etree.ElementTree as ET
import cv2


def download_and_extract_kemerovo(download_url: str, extract_to: str) -> None:
    """
    Download the KEMEROVO dataset zip file from the provided URL, extract it to the specified folder,
    rename the extracted folder "Stenosis detection" to "KEMEROVO", and remove extraneous files/folders.

    Args:
        download_url (str): URL to download the KEMEROVO zip file.
        extract_to (str): Directory where the dataset should be extracted.
                      For example, "/home/mario/Python/Datasets".
    """
    local_zip: str = os.path.join(extract_to, "kemerovo_dataset.zip")
    print(f"Downloading KEMEROVO dataset from {download_url} ...")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    with open(local_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded dataset to {local_zip}")

    print(f"Extracting {local_zip} ...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted dataset to {extract_to}")

    # Remove the zip file after extraction.
    os.remove(local_zip)

    # The extracted folder is named "Stenosis detection"; rename it to "KEMEROVO".
    extracted_folder: str = os.path.join(extract_to, "Stenosis detection")
    kemerovo_folder: str = os.path.join(extract_to, "KEMEROVO")
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, kemerovo_folder)
        print(f"Renamed folder to {kemerovo_folder}")
    else:
        print(f"Expected folder 'Stenosis detection' not found in {extract_to}")

    # Remove extraneous files and folders from KEMEROVO directory.
    cleanup_kemerovo_folder(kemerovo_folder)


def cleanup_kemerovo_folder(kemerovo_path: str) -> None:
    """
    Remove extraneous files and folders from the KEMEROVO directory.
    Only keep the "dataset" folder; delete any other files or folders.

    Args:
        kemerovo_path (str): Path to the KEMEROVO folder.
    """
    for item in os.listdir(kemerovo_path):
        item_path = os.path.join(kemerovo_path, item)
        if item != "dataset":
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Removed file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path}")


def convert_bbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_width: float,
    img_height: float,
) -> Dict[str, float]:
    """
    Convert bounding box from Pascal VOC format (xmin, ymin, xmax, ymax) to YOLO normalized format.

    Args:
        xmin (float): x-coordinate of the left side.
        ymin (float): y-coordinate of the top side.
        xmax (float): x-coordinate of the right side.
        ymax (float): y-coordinate of the bottom side.
        img_width (float): width of the image in pixels.
        img_height (float): height of the image in pixels.

    Returns:
        Dict[str, float]: A dictionary with keys "x_center", "y_center", "width", "height" (all normalized).
    """
    box_width: float = xmax - xmin
    box_height: float = ymax - ymin
    x_center: float = (xmin + xmax) / 2.0 / img_width
    y_center: float = (ymin + ymax) / 2.0 / img_height
    return {
        "x_center": x_center,
        "y_center": y_center,
        "width": box_width / img_width,
        "height": box_height / img_height,
    }


def parse_voc_xml(
    xml_file: str, image_path: str
) -> Tuple[Optional[str], Optional[int], Optional[int], List[Dict[str, Any]]]:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text
        size_elem = root.find("size")
        xml_width = int(size_elem.find("width").text)
        xml_height = int(size_elem.find("height").text)

        actual_img = cv2.imread(image_path)
        if actual_img is not None:
            h, w = actual_img.shape[:2]
            if (xml_width != w) or (xml_height != h):
                print(f"Warning: mismatch in {xml_file}")

        bboxes = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # In well-formed Pascal VOC, xmin < xmax, ymin < ymax
            # If there's a chance of corruption, you could sort them or clamp them
            # e.g. xmin, xmax = sorted((xmin, xmax)) and likewise for y

            bboxes.append(
                {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "label": label}
            )

        return filename, xml_width, xml_height, bboxes
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return None, None, None, []


def process_kemerovo_dataset(root_dir: str) -> Dict[str, Any]:
    """
    Process the KEMEROVO dataset and generate a standardized JSON structure.

    The KEMEROVO dataset is expected to be located at:
        root_dir/KEMEROVO/dataset/
    Each image has a corresponding annotation in Pascal VOC XML format.

    For each image:
      - Read the XML file to obtain image filename, resolution, and object annotations.
      - Convert each bounding box from [xmin, ymin, xmax, ymax] (pixel format) into YOLO normalized coordinates.
      - Build an "annotations" dictionary with a "name" (ending with ".txt") and keys "bbox1", "bbox2", ...
      - Set the "lesion" flag to True (as annotations are expected for each image).
      - Create an entry with a new standardized ID using the image filename. The filename follows the structure:
            14_{patient}_{video}_{frame}
        The unique id is then constructed as:
            kemerovo_p{patient}_v{video}_{frame.zfill(5)}

    Returns:
        Dict[str, Any]: A dictionary with the standardized JSON structure:
           {
             "Standard_dataset": {
                 "kemerovo_p002_v5_00016": { ... },
                 "kemerovo_pXXX_vYYY_ZZZZZ": { ... },
                 ...
             }
           }
    """
    standard_dataset: Dict[str, Any] = {}
    dataset_dir: str = os.path.join(root_dir, "KEMEROVO", "dataset")
    if not os.path.isdir(dataset_dir):
        print(f"Dataset folder not found: {dataset_dir}")
        return {"Standard_dataset": standard_dataset}

    # Iterate over XML files in the dataset folder.
    for file in os.listdir(dataset_dir):
        if file.lower().endswith(".xml"):
            xml_path: str = os.path.join(dataset_dir, file)

            # Determine corresponding image file (assume same base name, with .bmp extension)
            base_name: str = os.path.splitext(file)[0]  # e.g., "14_002_5_0016"
            image_filename: str = f"{base_name}.bmp"
            image_path: str = os.path.join(dataset_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Image file not found for {xml_path}: {image_path}")
                continue

            filename, img_width, img_height, bboxes = parse_voc_xml(
                xml_path, image_path
            )
            if filename is None:
                continue

            # Extract parts from the filename (expected: 14_{patient}_{video}_{frame})
            parts = base_name.split("_")
            if len(parts) < 4:
                print(f"Filename {base_name} does not follow expected pattern.")
                continue
            patient: str = parts[1]
            video: str = parts[2]
            frame: str = parts[3]
            std_id: str = f"kemerovo_p{patient}_v{video}_{frame.zfill(5)}"

            # Convert each bounding box to YOLO format.
            transformed_bboxes: List[Dict[str, Any]] = []
            for bbox in bboxes:
                conv_bbox: Dict[str, float] = convert_bbox_to_yolo(
                    bbox["xmin"],
                    bbox["ymin"],
                    bbox["xmax"],
                    bbox["ymax"],
                    img_width,
                    img_height,
                )
                conv_bbox["label"] = bbox["label"]
                transformed_bboxes.append(conv_bbox)
            # Build the annotations dictionary.
            annotations_dict: Dict[str, Any] = {"name": f"{std_id}.txt"}
            for idx, t_bbox in enumerate(transformed_bboxes, start=1):
                annotations_dict[f"bbox{idx}"] = t_bbox

            # For KEMEROVO, we assume a lesion is always present if an XML exists.
            lesion_flag: bool = True

            entry: Dict[str, Any] = {
                "id": std_id,
                "dataset_origin": "kemerovo",
                "lesion": lesion_flag,
                "image": {
                    "name": f"{std_id}.bmp",
                    "route": image_path,
                    "original_name": image_filename,
                    "height": img_height,
                    "width": img_width,
                },
                "annotations": annotations_dict,
            }
            standard_dataset[std_id] = entry

    return {"Standard_dataset": standard_dataset}


if __name__ == "__main__":
    # Example usage:
    # 1. Download, extract, and clean up the dataset.
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
    )
    extract_folder: str = "/home/mario/Python/Datasets"  # Adjust as required.
    # download_and_extract_kemerovo(download_url, extract_folder)

    # 2. Process the dataset.
    json_data: Dict[str, Any] = process_kemerovo_dataset(root_dir=extract_folder)
    output_json_file: str = "./kemerovo_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
