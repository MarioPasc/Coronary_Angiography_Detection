"""
dca_yolo_preprocessing
======================

A tiny, dependency-free preprocessing pipeline for YOLO (or any CV task)
that performs Canny edge detection, histogram equalisation and weighted
image fusion.

Public API
----------
preprocess(image, *, low_threshold=10, high_threshold=35,
           alpha=0.3, blur_kernel=(5, 5)) -> numpy.ndarray

Everything else is intentionally private.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

__all__ = ["preprocess"]  # the *only* symbol we export





# -----------------------------------------------------------------------------#
#                               Private helpers                                #
# -----------------------------------------------------------------------------#
_ImageLike = Union[str, Path, np.ndarray]


def _read_as_gray(img: _ImageLike) -> np.ndarray:
    """
    Load an image from path or return the numpy array as-is, then convert to
    single-channel uint8 grayscale.

    Parameters
    ----------
    img
        Either a file path / Path object or an already-loaded OpenCV / NumPy
        image.

    Returns
    -------
    gray : numpy.ndarray
        8-bit, single-channel image.
    """
    if isinstance(img, (str, Path)):
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)  # type: ignore[arg-type]
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {img}")
        return img
    if img.ndim == 2:  # already gray
        return img.astype(np.uint8)
    # colour → gray
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _apply_canny(
    gray: np.ndarray,
    low_threshold: int,
    high_threshold: int,
    blur_kernel: Tuple[int, int],
) -> np.ndarray:
    """
    Gaussian-smooth then run Canny edge detector.

    Returns
    -------
    edges : numpy.ndarray
        Binary edge map (0 / 255, uint8).
    """
    blurred: np.ndarray = cv2.GaussianBlur(gray, blur_kernel, sigmaX=0)
    edges: np.ndarray = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def _hist_equalisation(gray: np.ndarray) -> np.ndarray:
    """
    Contrast enhancement with classical histogram equalisation (OpenCV impl.).
    """
    return cv2.equalizeHist(gray)


def _fuse_images(
    equalised: np.ndarray, edges: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Weighted pixel-wise fusion: F = clip(equalised + alpha * edges).
    """
    fused = equalised.astype(np.float32) + alpha * edges.astype(np.float32)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


# -----------------------------------------------------------------------------#
#                               Public pipeline                                #
# -----------------------------------------------------------------------------#
def preprocess(
    image: _ImageLike,
    *,
    low_threshold: int = 10,
    high_threshold: int = 35,
    alpha: float = 0.3,
    blur_kernel: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """
    Run *Canny → Histogram-Equalisation → Fusion* on **one** image.

    Parameters
    ----------
    image
        Either an image file path or an already-loaded `np.ndarray`.
    low_threshold, high_threshold
        Canny hysteresis thresholds (intensity units).
    alpha
        Fusion weight in ``result = equalised + alpha * edges``.
    blur_kernel
        (kH, kW) odd kernel size for the Gaussian pre-blur inside Canny.

    Returns
    -------
    fused : numpy.ndarray
        Final 8-bit single-channel image, ready to be fed to a detector
        or saved with `cv2.imwrite`.
    """
    gray = _read_as_gray(image)
    edges = _apply_canny(gray, low_threshold, high_threshold, blur_kernel)
    equalised = _hist_equalisation(gray)
    fused = _fuse_images(equalised, edges, alpha)
    return fused


def apply_dca_yolo_preprocessing(
    image_path: str | Path,
    output_path: str | Path,
    low_threshold: int = 10,
    high_threshold: int = 35,
    alpha: float = 0.30,
    blur_kernel: Tuple[int, int] = (5, 5),
) -> bool:
    """Run the three-stage pipeline and write the result to *output_path*."""
    try:
        fused = preprocess(
            image_path,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            alpha=alpha,
            blur_kernel=blur_kernel,
        )
        cv2.imwrite(str(output_path), fused)
        return True
    except Exception as exc:           # noqa: BLE001
        print(f"[DCA-YOLO] Error processing {image_path}: {exc}")
        return False


# -----------------------------------------------------------------------------#
#                           Optional CLI / quick test                          #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DCA-YOLO preprocessing on an image."
    )
    parser.add_argument("input", help="Path to input image.")
    parser.add_argument(
        "-o", "--output", default="preprocessed.png", help="Output filename"
    )
    parser.add_argument("--low", type=int, default=10, help="Canny low threshold")
    parser.add_argument("--high", type=int, default=35, help="Canny high threshold")
    parser.add_argument("--alpha", type=float, default=0.3, help="Fusion weight")
    args = parser.parse_args()

    out = preprocess(
        args.input, low_threshold=args.low, high_threshold=args.high, alpha=args.alpha
    )
    cv2.imwrite(args.output, out)
    print(f"Saved ✅  {args.output}")
