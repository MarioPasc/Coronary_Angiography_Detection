from typing import Optional
import os
import re
import logging
from PIL import Image
from pathlib import Path


def find_mask_path(masks_folder: str, img_basename: str) -> Optional[str]:
    """
    Find the corresponding mask file for an image.

    Args:
        masks_folder: Directory containing mask files
        img_basename: Base filename of the input image

    Returns:
        Full path to the mask file or None if not found
    """
    logging.info(f"Looking for mask with basename: {img_basename}")

    try:
        mask_files = os.listdir(masks_folder)

        # Try exact match
        for mask_file in mask_files:
            mask_basename = Path(mask_file).stem
            if mask_basename == img_basename:
                mask_path = os.path.join(masks_folder, mask_file)
                logging.info(f"Found exact match mask: {mask_path}")
                return mask_path

        # Try with different extensions
        for mask_file in mask_files:
            mask_basename = Path(mask_file).stem
            if mask_basename.startswith(img_basename):
                mask_path = os.path.join(masks_folder, mask_file)
                logging.info(f"Found partial match mask: {mask_path}")
                return mask_path

        # Try to extract numerical part
        number_match = re.search(r"(\d+)$", img_basename)
        if number_match:
            number = number_match.group(1)
            for mask_file in mask_files:
                if number in Path(mask_file).stem:
                    mask_path = os.path.join(masks_folder, mask_file)
                    logging.info(f"Found number match mask: {mask_path}")
                    return mask_path

        logging.error(f"No matching mask found for {img_basename}")
        return None

    except Exception as e:
        logging.error(f"Error searching for mask: {e}")
        return None
