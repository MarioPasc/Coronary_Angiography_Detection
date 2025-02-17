# ica_yolo_detection/datasets/dataset1.py

from typing import List, Dict, Optional, Tuple, Any
import os
import requests
import zipfile
import json
from PIL import Image


def download_and_extract_cadica(download_url: str, extract_to: str) -> None:
    """
    Download the CADICA dataset from the provided URL and extract it to the specified folder.

    Args:
        download_url (str): URL to download the CADICA zip file.
        extract_to (str): Directory where the dataset should be extracted.
    """
    local_zip: str = os.path.join(extract_to, "cadica_dataset.zip")
    print(f"Downloading CADICA dataset from {download_url} ...")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()  # Catch HTTP errors
    with open(local_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded dataset to {local_zip}")

    print(f"Extracting {local_zip} ...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted dataset to {extract_to}")

    # Optionally, remove the zip file after extraction
    os.remove(local_zip)


def read_selected_frames(file_path: str) -> List[str]:
    """
    Read a selected frames text file and return a list of frame identifiers.

    Args:
        file_path (str): Path to the selected frames text file.

    Returns:
        List[str]: List of frame IDs.
    """
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
    """
    Parse multiple bounding boxes and labels from an annotation file.

    Each line in the file should have values in the format:
        x y w h label

    Args:
        annotation_file_path (str): Path to the annotation text file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each with keys "X", "Y", "W", "H", and "label".
    """
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


def process_video(
    video_dir: str, patient_id: str, lesion_flag: bool
) -> Dict[str, Any]:
    """
    Process a single video directory for a given patient.

    It reads the selected frames file, locates the corresponding image files in the 'input'
    folder, and (if available and lesion_flag is True) the annotation files in the 'groundtruth' folder.
    The unique id for each image is constructed using the dataset name, patient, video, and frame:
        "cadica_{patient}_{video}_{frame}"
    The function also converts the CADICA pixel bounding box coordinates (x, y, w, h) into YOLO
    normalized coordinates (x_center, y_center, width, height) using the image's resolution.

    Args:
        video_dir (str): Path to the video directory (e.g., ".../p11/v10").
        patient_id (str): The patient folder name (e.g., "p11").
        lesion_flag (bool): True if this video has lesion annotations; False if it's a control video.

    Returns:
        Dict[str, Any]: A dictionary mapping unique ids to JSON entries for this video.
    """
    entries: Dict[str, Any] = {}
    video_id: str = os.path.basename(video_dir)

    # Construct the selected frames file name (e.g., "p11_v10_selectedFrames.txt")
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
        # Construct the full image filename.
        # If the selected frame does not already include patient and video info,
        # then combine them. Otherwise, assume it's in the format "p{patient}_v{video}_{frame}".
        if frame.startswith(patient_id + "_" + video_id + "_"):
            full_frame: str = frame
        else:
            full_frame = f"{patient_id}_{video_id}_{frame}"
        image_filename: str = f"{full_frame}.png"
        image_path: str = os.path.join(input_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        # Open image to get its resolution.
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Construct unique id using dataset, patient, video and frame.
        unique_id: str = f"cadica_{full_frame}"

        # Parse bounding boxes (in pixel coordinates) if lesion annotations are expected.
        bboxes: List[Dict[str, Any]] = []
        if lesion_flag and has_groundtruth:
            annotation_filename: str = f"{full_frame}.txt"
            annotation_path: str = os.path.join(groundtruth_dir, annotation_filename)
            bboxes = parse_bounding_boxes(annotation_path)

        # Transform each bounding box into YOLO normalized format.
        transformed_bboxes: List[Dict[str, Any]] = []
        for bbox in bboxes:
            try:
                x = bbox["X"]
                y = bbox["Y"]
                w_pixel = bbox["W"]
                h_pixel = bbox["H"]
                # Compute center coordinates (in pixels) and normalize them.
                x_center = (x + w_pixel / 2) / img_width
                y_center = (y + h_pixel / 2) / img_height
                w_norm = w_pixel / img_width
                h_norm = h_pixel / img_height
                transformed_bbox = {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": w_norm,
                    "height": h_norm,
                    "label": bbox["label"],
                }
                transformed_bboxes.append(transformed_bbox)
            except Exception as e:
                print(f"Error transforming bbox {bbox} due to {e}")

        # Build the annotations dictionary.
        annotations: Dict[str, Any] = {"name": f"{unique_id}.txt"}
        for idx, t_bbox in enumerate(transformed_bboxes, start=1):
            annotations[f"bbox{idx}"] = t_bbox

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
    """
    Process the CADICA dataset given the root directory of the unzipped dataset.

    Expected directory structure:

        root_dir/
            metadata.xlsx
            nonselectedVideos/
            readme.txt
            selectedVideos/
                pX/                 <-- patient folders (e.g., "p11")
                    lesionVideos.txt
                    nonlesionVideos.txt
                    vY/             <-- video folders (e.g., "v10")
                        input/      <-- contains PNG image frames
                        groundtruth/ (optional) <-- contains annotation files and a groundTruthTable.mat
                        pX_vY_selectedFrames.txt  <-- list of keyframe IDs

    Returns:
        Dict[str, Any]: A dictionary with the standardized JSON structure:
            {
              "Standard_dataset": {
                  "cadica_p11_v10_00026": { ... },
                  "cadica_p11_v10_00027": { ... },
                  ...
              }
            }
    """
    standard_dataset: Dict[str, Any] = {}
    selected_videos_dir: str = os.path.join(root_dir, "selectedVideos")
    if not os.path.isdir(selected_videos_dir):
        print(f"Selected videos directory not found in {root_dir}")
        return {"Standard_dataset": standard_dataset}

    # Iterate over patient directories.
    for patient in os.listdir(selected_videos_dir):
        patient_dir: str = os.path.join(selected_videos_dir, patient)
        if not os.path.isdir(patient_dir) or not patient.lower().startswith("p"):
            continue

        # Read lesion and non-lesion video lists.
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

        # Iterate over video directories within each patient folder.
        for video in os.listdir(patient_dir):
            # Skip non-directory entries.
            if not video.lower().startswith("v"):
                continue
            video_dir: str = os.path.join(patient_dir, video)
            if not os.path.isdir(video_dir):
                continue

            # Determine if this video is a lesion video or not.
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
    # Example usage:
    # 1. Download and extract the dataset.
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/p9bpx9ctcv/files/3d00fcc4-e555-47e8-bf7d-fa39aa4bf56e/file_downloaded"
    )
    extract_folder: str = "/home/mario/Python/Datasets"  # Adjusted folder path as required
    # download_and_extract_cadica(download_url, extract_folder)

    # 2. Process the dataset.
    # Assuming the extracted structure is: extract_folder/CADICA/CADICA/
    dataset_root: str = os.path.join(extract_folder, "CADICA")
    json_data: Dict[str, Any] = process_cadica_dataset(dataset_root)

    # 3. Save the standardized JSON to a file.
    output_json_file: str = "cadica_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
