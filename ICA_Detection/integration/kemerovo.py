import os
import shutil
import requests
import zipfile
import json
from typing import List, Dict, Tuple, Any, Optional
import xml.etree.ElementTree as ET
import cv2
from ICA_Detection.tools.bbox_translation import kemerovo_to_common


def download_and_extract_kemerovo(download_url: str, extract_to: str) -> None:
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
    os.remove(local_zip)
    extracted_folder: str = os.path.join(extract_to, "Stenosis detection")
    kemerovo_folder: str = os.path.join(extract_to, "KEMEROVO")
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, kemerovo_folder)
        print(f"Renamed folder to {kemerovo_folder}")
    else:
        print(f"Expected folder 'Stenosis detection' not found in {extract_to}")
    cleanup_kemerovo_folder(kemerovo_folder)


def cleanup_kemerovo_folder(kemerovo_path: str) -> None:
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
    # This function is replaced by our translator.
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
        filename: str = root.find("filename").text  # type: ignore
        size_elem = root.find("size")
        img_width: int = int(size_elem.find("width").text)  # type: ignore
        img_height: int = int(size_elem.find("height").text)  # type: ignore
        actual_img = cv2.imread(image_path)
        if actual_img is not None:
            h, w = actual_img.shape[:2]
            if (img_width != w) or (img_height != h):
                print(f"Warning: mismatch in {xml_file}")
        bboxes: List[Dict[str, Any]] = []
        for obj in root.findall("object"):
            label = obj.find("name").text  # type: ignore
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)  # type: ignore
            ymin = float(bndbox.find("ymin").text)  # type: ignore
            xmax = float(bndbox.find("xmax").text)  # type: ignore
            ymax = float(bndbox.find("ymax").text)  # type: ignore
            bboxes.append(
                {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "label": label}
            )
        return filename, img_width, img_height, bboxes
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return None, None, None, []


def process_kemerovo_dataset(root_dir: str) -> Dict[str, Any]:
    standard_dataset: Dict[str, Any] = {}
    dataset_dir: str = os.path.join(root_dir, "KEMEROVO", "dataset")
    if not os.path.isdir(dataset_dir):
        print(f"Dataset folder not found: {dataset_dir}")
        return {"Standard_dataset": standard_dataset}
    for file in os.listdir(dataset_dir):
        if file.lower().endswith(".xml"):
            xml_path: str = os.path.join(dataset_dir, file)
            base_name: str = os.path.splitext(file)[0]
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
            parts = base_name.split("_")
            if len(parts) < 4:
                print(f"Filename {base_name} does not follow expected pattern.")
                continue
            patient: str = parts[1]
            video: str = parts[2]
            frame: str = parts[3]
            std_id: str = f"kemerovo_p{patient}_v{video}_{frame.zfill(5)}"
            transformed_bboxes: List[Dict[str, Any]] = []

            for bbox in bboxes:
                common_bbox = kemerovo_to_common(bbox)

                transformed_bboxes.append(common_bbox)
            annotations_dict: Dict[str, Any] = {"name": f"{std_id}.txt"}
            for idx, t_bbox in enumerate(transformed_bboxes, start=1):
                annotations_dict[f"bbox{idx}"] = t_bbox
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
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
    )
    extract_folder: str = "/home/mario/Python/Datasets"
    # download_and_extract_kemerovo(download_url, extract_folder)
    json_data: Dict[str, Any] = process_kemerovo_dataset(root_dir=extract_folder)
    output_json_file: str = "./kemerovo_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
