from dataclasses import dataclass
from typing import Union, Dict, Any, List


@dataclass
class MaskGuidedAttentionConfig:
    """Configuration for Mask-Guided Attention training."""

    model_cfg: str
    data_yaml: str
    masks_folder: str
    epochs: int = 100
    imgsz: int = 640
    visualize_interval: int = 100
    target_layers: Union[List[str], None] = None
    device: str = "cuda:0"
    alpha: float = 0
    batch: int = 4

    # Training configuration
    project_dir: str = "./runs/train"
    experiment_name: str = "mga_yolo"

    # Augmentation settings (all disabled by default for MGA)
    augmentation_config: Union[Dict[str, Any], None] = None

    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.target_layers is None:
            self.target_layers = [
                "15",
                "18",
                "21",  # LOS PUTOS NOMBRE AJHAJJA
            ]  # P3, P4, P5 features

        if self.augmentation_config is None:
            self.augmentation_config = {
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "bgr": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "auto_augment": None,
                "erasing": 0.0,
                "crop_fraction": 0.0,
            }
