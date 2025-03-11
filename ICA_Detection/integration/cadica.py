import os
import requests
import zipfile
import json
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image

# Import translator functions for Cadica.
from ICA_Detection.tools.bbox_translation import cadica_to_common


def download_and_extract_cadica(download_url: str, extract_to: str) -> None:
    # (Same as before)
    local_zip: str = os.path.join(extract_to, "cadica_dataset.zip")
    print(f"Downloading CADICA dataset from {download_url} ...")
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


def read_selected_frames(file_path: str) -> List[str]:
    frames: List[str] = []
    if not os.path.exists(file_path):
        print(f"Selected frames file not found: {file_path}")
        return frames
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(line)
    return frames


def parse_bounding_boxes(annotation_file_path: str) -> List[Dict[str, Any]]:
    bboxes: List[Dict[str, Any]] = []
    if not os.path.exists(annotation_file_path):
        return bboxes
    try:
        with open(annotation_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            bbox = {
                                "X": float(parts[0]),
                                "Y": float(parts[1]),
                                "W": float(parts[2]),
                                "H": float(parts[3]),
                                "label": parts[4],
                            }
                            bboxes.append(bbox)
                        except ValueError:
                            print(
                                f"Error parsing bounding box values in {annotation_file_path}"
                            )
                    else:
                        print(
                            f"Unexpected format in annotation file: {annotation_file_path}"
                        )
    except Exception as e:
        print(f"Error reading annotation file {annotation_file_path}: {e}")
    return bboxes


def process_video(video_dir: str, patient_id: str, lesion_flag: bool) -> Dict[str, Any]:
    entries: Dict[str, Any] = {}
    video_id: str = os.path.basename(video_dir)
    selected_frames_file: str = os.path.join(
        video_dir, f"{patient_id}_{video_id}_selectedFrames.txt"
    )
    selected_frames: List[str] = read_selected_frames(selected_frames_file)
    if not selected_frames:
        print(f"No selected frames found in {selected_frames_file}")
        return entries
    input_dir: str = os.path.join(video_dir, "input")
    groundtruth_dir: str = os.path.join(video_dir, "groundtruth")
    has_groundtruth: bool = os.path.isdir(groundtruth_dir)
    for frame in selected_frames:
        if frame.startswith(patient_id + "_" + video_id + "_"):
            full_frame: str = frame
        else:
            full_frame = f"{patient_id}_{video_id}_{frame}"
        image_filename: str = f"{full_frame}.png"
        image_path: str = os.path.join(input_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        unique_id: str = f"cadica_{full_frame}"
        bboxes: List[Dict[str, Any]] = []
        if lesion_flag and has_groundtruth:
            annotation_filename: str = f"{full_frame}.txt"
            annotation_path: str = os.path.join(groundtruth_dir, annotation_filename)
            bboxes = parse_bounding_boxes(annotation_path)
        transformed_bboxes = []
        for bbox in bboxes:
            # Convert native Cadica bbox to common format, then to YOLO.
            common_bbox = cadica_to_common(bbox)

            transformed_bboxes.append(common_bbox)
        annotations: Dict[str, Any] = {"name": f"{unique_id}.txt"}
        stenosis_dict: Dict[str, Any] = {}
        for idx, t_bbox in enumerate(transformed_bboxes, start=1):
            stenosis_dict[f"bbox{idx}"] = t_bbox
        annotations["stenosis"] = stenosis_dict

        entry: Dict[str, Any] = {
            "id": unique_id,
            "dataset_origin": "cadica",
            "lesion": lesion_flag,
            "image": {
                "name": f"{unique_id}.png",
                "route": image_path,
                "original_name": image_filename,
                "height": img_height,
                "width": img_width,
            },
            "annotations": annotations,
        }
        entries[unique_id] = entry
    return entries


def process_cadica_dataset(root_dir: str) -> Dict[str, Any]:
    root_dir = os.path.join(root_dir, "CADICA")
    standard_dataset: Dict[str, Any] = {}
    selected_videos_dir: str = os.path.join(root_dir, "selectedVideos")
    if not os.path.isdir(selected_videos_dir):
        print(f"Selected videos directory not found in {root_dir}")
        return {"Standard_dataset": standard_dataset}
    for patient in os.listdir(selected_videos_dir):
        patient_dir: str = os.path.join(selected_videos_dir, patient)
        if not os.path.isdir(patient_dir) or not patient.lower().startswith("p"):
            continue
        lesion_videos: set = set()
        nonlesion_videos: set = set()
        lesion_file: str = os.path.join(patient_dir, "lesionVideos.txt")
        nonlesion_file: str = os.path.join(patient_dir, "nonlesionVideos.txt")
        if os.path.exists(lesion_file):
            with open(lesion_file, "r") as f:
                for line in f:
                    video_name = line.strip()
                    if video_name:
                        lesion_videos.add(video_name)
        if os.path.exists(nonlesion_file):
            with open(nonlesion_file, "r") as f:
                for line in f:
                    video_name = line.strip()
                    if video_name:
                        nonlesion_videos.add(video_name)
        for video in os.listdir(patient_dir):
            if not video.lower().startswith("v"):
                continue
            video_dir: str = os.path.join(patient_dir, video)
            if not os.path.isdir(video_dir):
                continue
            if video in lesion_videos:
                lesion_flag: bool = True
            elif video in nonlesion_videos:
                lesion_flag = False
            else:
                print(
                    f"Video {video} in {patient} not listed in lesion or nonlesion files. Skipping."
                )
                continue
            entries = process_video(video_dir, patient, lesion_flag)
            standard_dataset.update(entries)
    return {"Standard_dataset": standard_dataset}


if __name__ == "__main__":
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/p9bpx9ctcv/files/3d00fcc4-e555-47e8-bf7d-fa39aa4bf56e/file_downloaded"
    )
    extract_folder: str = "/home/mario/Python/Datasets"
    # download_and_extract_cadica(download_url, extract_folder)
    dataset_root: str = os.path.join(extract_folder, "CADICA")
    json_data: Dict[str, Any] = process_cadica_dataset(dataset_root)
    output_json_file: str = "cadica_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
