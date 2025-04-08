import os
import torch
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

# Import relevant modules from the codebase
from ICA_Detection.external.ultralytics.ultralytics import YOLO
from ICA_Detection.tasks.mga_yolo.models.hooks import HookManager
from ICA_Detection.tasks.mga_yolo.cfg.defaults import MaskGuidedAttentionConfig


class MaskGuidedInference:
    """
    Performs inference with a mask-guided attention YOLO model.

    This class applies a provided mask to guide the model's attention
    during inference, similar to how it was trained with MGA.
    """

    def __init__(
        self,
        config: MaskGuidedAttentionConfig,
    ) -> None:
        """
        Initialize the mask-guided inference model.

        Args:
            config: Configuration object for MGA inference
        """
        self.config = config
        self.model_path = config.model_cfg
        self.target_layers = config.target_layers
        self.alpha = config.alpha
        self.imgsz = config.imgsz
        self.device = config.device

        # Initialize the model
        self.model = YOLO(self.model_path)
        self.current_mask_path: Optional[str] = None
        self.hook_manager = None

        # Print distinctive banner
        self._print_mga_banner()

        logging.info(f"MGA-YOLO Inference: Using model {self.model_path}")
        logging.info(f"MGA-YOLO Inference: Target layers {self.target_layers}")
        logging.info(f"MGA-YOLO Inference: Alpha value {self.alpha}")

    def _print_mga_banner(self) -> None:
        """Print a distinctive banner to mark MGA-YOLO inference."""
        banner = """
        ╔═══════════════════════════════════════════════╗
        ║                                               ║
        ║    MGA-YOLO: Mask-Guided Attention YOLO       ║
        ║              INFERENCE MODE                   ║
        ║                                               ║
        ╚═══════════════════════════════════════════════╝
        """
        print(banner)

    def _setup_hooks(self, mask_path: str) -> None:
        """
        Set up hooks for mask-guided inference.

        Args:
            mask_path: Path to the mask file to use during inference
        """
        # Store mask path and its directory
        self.current_mask_path = mask_path
        mask_folder = str(Path(mask_path).parent)

        # Create a function that returns our image path for any batch index
        def get_image_path_fn(batch_idx: int) -> Optional[str]:
            return "current_image.jpg"  # Placeholder name

        # Initialize hook manager
        self.hook_manager = HookManager(
            masks_folder=mask_folder,
            target_layers=self.target_layers,
            get_image_path_fn=get_image_path_fn,
            alpha=self.alpha,
        )

        # Override the mask finding function to always return our specific mask
        original_find_mask = self.hook_manager._find_mask_path

        def find_mask_path_override(img_basename: str) -> Optional[str]:
            return self.current_mask_path

        self.hook_manager._find_mask_path = find_mask_path_override

        # Register hooks to model
        self.model = self.hook_manager.register_hooks(self.model)

        # Set batch paths (needed for hook manager to function)
        self.hook_manager.set_batch_paths(["current_image.jpg"])

        logging.info(f"MGA-YOLO Inference: Using mask {mask_path}")
        logging.info("MGA-YOLO Inference: Hooks successfully registered")

    def predict(
        self,
        image_path: str,
        mask_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        show: bool = True,
        save: bool = False,
        save_dir: Optional[str] = None,
    ) -> Any:
        """
        Run inference on an image with mask guidance.

        Args:
            image_path: Path to input image
            mask_path: Path to mask file
            conf: Confidence threshold for detections
            iou: IoU threshold for NMS
            show: Whether to display the results
            save: Whether to save the results to disk
            save_dir: Directory to save results (if save=True)

        Returns:
            Object containing inference results
        """
        # Set up hooks with the provided mask
        self._setup_hooks(mask_path)

        logging.info(f"MGA-YOLO Inference: Running on {image_path}")

        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            show=show,
            save=save,
            project=save_dir if save and save_dir else None,
            imgsz=self.imgsz,
            device=self.device,
        )

        # Clean up hooks after inference to prevent memory leaks
        if self.hook_manager:
            self.hook_manager.clear_hooks()

        logging.info("MGA-YOLO Inference: Prediction complete")

        return results[0]  # Return results from first (only) image

    def visualize_results(
        self,
        results: Any,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        Visualize detection results on the image.

        Args:
            results: Results object from YOLO prediction
            output_path: Path to save the visualization (optional)
            show: Whether to display the visualization

        Returns:
            Annotated image as numpy array
        """
        # Let the YOLO model handle plotting
        annotated_img = results.plot()

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_img)
            logging.info(f"MGA-YOLO Inference: Results saved to {output_path}")

        # Show if requested
        if show:
            cv2.imshow("MGA-YOLO Results", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_img


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    base = "/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA"
    # Define parameters
    model_path = os.path.join(
        base, "runs/train/mga_yolo_paper_skipconnection/weights/best.pt"
    )
    alpha = -1  # Skip connection alpha value

    # Image and mask paths
    image_path = os.path.join(
        base, "detection", "images", "val", "arcadetest_p66_v66_00066.png"
    )
    mask_path = os.path.join(base, "masks", "arcadetest_p66_v66_00066.png")

    # Define MGA configuration
    mga_config = MaskGuidedAttentionConfig(
        model_cfg="yolov8n.pt",
        data_yaml="/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/detection/yolo_ica_detection.yaml",
        masks_folder=os.path.join(base, "masks"),
        alpha=alpha,  # Skip connection alpha value
        imgsz=512,
        device="cuda:0",
    )

    # Initialize inference with config
    mga_inference = MaskGuidedInference(config=mga_config)

    # Run inference
    results = mga_inference.predict(
        image_path=image_path,
        mask_path=mask_path,
        conf=0.25,
        iou=0.45,
        show=False,  # Don't show during prediction
        save=False,  # Don't save automatically
    )

    # Print detection information
    print(f"Found {len(results.boxes)} objects")
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        confidence = box.conf.item()
        print(f"Object {i+1}: Class {cls_id}, Confidence: {confidence:.4f}")

    # Visualize results
    mga_inference.visualize_results(
        results=results, output_path="mga_yolo_result.jpg", show=True
    )


if __name__ == "__main__":
    main()
