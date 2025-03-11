# ica_yolo_detection/tools/bbox_translators.py

"""
Common Format:
A dictionary with keys:

    "xmin": The x-coordinate (in pixels) of the top-left corner
    "ymin": The y-coordinate (in pixels) of the top-left corner
    "xmax": The x-coordinate (in pixels) of the bottom-right corner
    "ymax": The y-coordinate (in pixels) of the bottom-right corner
    "label": The associated label

This format is essentially the Pascal VOC format in pixel coordinates.
It is independent of image dimensions and can easily be translated into
other formats (e.g., normalized YOLO format) when needed.
"""

from typing import Dict, Any
import numpy as np


def cadica_to_common(bbox: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Cadica bounding box to the common intermediate format.

    Cadica annotations are provided in the format:
      { "X": x, "Y": y, "W": w, "H": h, "label": label }
    where (x, y) is the top-left corner.

    Returns a dictionary with keys:
      "xmin", "ymin", "xmax", "ymax", "label"
    """
    x = bbox["X"]
    y = bbox["Y"]
    w = bbox["W"]
    h = bbox["H"]
    return {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h, "label": bbox["label"]}


def arcade_to_common(bbox: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an Arcade bounding box to the common intermediate format.

    Arcade annotations are provided in the "XYWH" format (x, y, w, h, label),
    where (x, y) is the top-left corner.

    Returns a dictionary with keys:
      "xmin", "ymin", "xmax", "ymax", "label"
    """
    x = bbox["x"] if "x" in bbox else bbox.get("X")
    y = bbox["y"] if "y" in bbox else bbox.get("Y")
    w = bbox["w"] if "w" in bbox else bbox.get("W")
    h = bbox["h"] if "h" in bbox else bbox.get("H")
    return {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h, "label": bbox["label"]}  # type: ignore


def kemerovo_to_common(bbox: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Kemerovo bounding box to the common intermediate format.

    Kemerovo annotations (from Pascal VOC XML) are already in the format:
      { "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "label": label }

    Returns the bounding box unchanged (optionally verifying required keys).
    """
    # Ensure all required keys are present.
    required_keys = ["xmin", "ymin", "xmax", "ymax", "label"]
    for key in required_keys:
        if key not in bbox:
            raise ValueError(f"Key {key} not found in Kemerovo bbox: {bbox}")
    return bbox


def common_to_yolo(
    bbox: Dict[str, Any], img_width: float, img_height: float
) -> Dict[str, float]:
    """
    Convert a bounding box in the common intermediate format (pixel coordinates)
    to YOLO normalized format.

    The common format is:
      { "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "label": label }

    The YOLO normalized format returns:
      { "x_center": x_center, "y_center": y_center, "width": width, "height": height, "label": label }

    Args:
        bbox (Dict[str, Any]): Bounding box in common format.
        img_width (float): Image width in pixels.
        img_height (float): Image height in pixels.

    Returns:
        Dict[str, float]: Bounding box in YOLO normalized format.
    """
    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]
    w_pixel = xmax - xmin
    h_pixel = ymax - ymin
    x_center = (xmin + w_pixel / 2) / img_width
    y_center = (ymin + h_pixel / 2) / img_height
    return {
        "x_center": x_center,
        "y_center": y_center,
        "width": w_pixel / img_width,
        "height": h_pixel / img_height,
        "label": bbox["label"],
    }


def rescale_bbox(bbox, orig_width, orig_height, new_width, new_height):
    """
    Rescale a Pascal VOC bounding box given the old and new image dimensions.

    Args:
        bbox (dict): Bounding box with keys "xmin", "ymin", "xmax", "ymax".
        orig_width (int): Original image width.
        orig_height (int): Original image height.
        new_width (int): New image width.
        new_height (int): New image height.

    Returns:
        dict: New bounding box with updated coordinates.
    """
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    return {
        "xmin": np.round(bbox["xmin"] * scale_x, 0),
        "ymin": np.round(bbox["ymin"] * scale_y, 0),
        "xmax": np.round(bbox["xmax"] * scale_x, 0),
        "ymax": np.round(bbox["ymax"] * scale_y, 0),
        "label": bbox.get("label", ""),
    }


if __name__ == "__main__":
    # Example usage:
    cadica_bbox = {"X": 100, "Y": 150, "W": 50, "H": 80, "label": "lesion"}
    arcade_bbox = {"x": 120, "y": 160, "w": 40, "h": 70, "label": "lesion"}
    kemerovo_bbox = {
        "xmin": 90,
        "ymin": 140,
        "xmax": 140,
        "ymax": 210,
        "label": "lesion",
    }

    common_cadica = cadica_to_common(cadica_bbox)
    common_arcade = arcade_to_common(arcade_bbox)
    common_kemerovo = kemerovo_to_common(kemerovo_bbox)

    print("Common Cadica:", common_cadica)
    print("Common Arcade:", common_arcade)
    print("Common Kemerovo:", common_kemerovo)

    # Convert one common bbox to YOLO normalized (assuming image size 800x800).
    yolo_bbox = common_to_yolo(common_cadica, 800, 800)
    print("YOLO normalized from Cadica:", yolo_bbox)


def calculate_polygon_area(points):
    """
    Calculate the area of a polygon given its vertices.
    Uses the Shoelace formula (also known as the Surveyor's formula).

    Args:
        points: List of coordinates [x1, y1, x2, y2, ...]

    Returns:
        Area of the polygon
    """
    # Convert flat list to pairs
    vertices = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

    # Apply Shoelace formula
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0

    return area


def calculate_bbox_from_segmentation(points):
    """
    Calculate bounding box from segmentation points in format [x1, y1, x2, y2, ...]

    Returns:
        [x_min, y_min, width, height] as required by COCO
    """
    # Extract x and y coordinates
    x_coords = points[0::2]
    y_coords = points[1::2]

    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min

    return [x_min, y_min, width, height]
