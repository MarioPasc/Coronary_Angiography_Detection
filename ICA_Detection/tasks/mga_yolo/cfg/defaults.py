from dataclasses import dataclass, field, asdict
from typing import Literal, Dict, Any, List, Optional, ClassVar
import yaml
from pathlib import Path


@dataclass
class MaskGuidedAttentionConfig:
    """Configuration for Mask-Guided Attention training."""

    # Model and data parameters
    model_cfg: str
    data_yaml: str
    masks_folder: str

    # Basic training parameters
    epochs: int = 100
    imgsz: int = 640
    batch: int = 4
    device: str = "cuda:0"

    # MGA specific parameters
    target_layers: List[str] = field(default_factory=lambda: ["15", "18", "21"])
    reduction_ratio: int = 16
    kernel_size: int = 7
    sam_cam_fusion: Literal["sequential", "concat", "add"] = "add"
    mga_pyramid_fusion: Literal["multiply", "add"] = "add"

    # Experiment tracking
    project_dir: str = "./runs/train"
    experiment_name: str = "mga_yolo"
    save_period: int = 10  # Save checkpoint every N epochs

    # Visualization settings
    visualize_features: bool = True
    visualize_path: Optional[str] = None

    # Advanced training settings
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Augmentation settings
    augmentation_config: Dict[str, Any] = field(default_factory=dict)

    # Class vars for schema validation
    _required_fields: ClassVar[List[str]] = ["model_cfg", "data_yaml", "masks_folder"]

    def __post_init__(self):
        """Initialize default values after initialization."""
        # Set default augmentation config if not provided
        if not self.augmentation_config:
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

        # Set default visualization path if enabled but not specified
        if self.visualize_features and self.visualize_path is None:
            self.visualize_path = str(
                Path(self.project_dir) / self.experiment_name / "visualizations"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MaskGuidedAttentionConfig":
        """
        Create configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            Configuration object populated from YAML file
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Validate required fields
        missing_fields = [
            field for field in cls._required_fields if field not in config_dict
        ]
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {missing_fields}")

        # Handle nested augmentation_config
        aug_config = config_dict.pop("augmentation_config", {})

        # Create config instance
        config = cls(**config_dict)

        # Apply augmentation config if provided
        if aug_config:
            config.augmentation_config = aug_config

        return config

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration
        """
        # Convert dataclass to dict
        config_dict = asdict(self)

        # Save to YAML
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def create_template(cls, yaml_path: str) -> None:
        """
        Create a template YAML configuration file.

        Args:
            yaml_path: Path where to save the template YAML
        """
        # Create a default instance
        default_config = cls(
            model_cfg="yolov8n.pt", data_yaml="data/custom.yaml", masks_folder="masks"
        )
        default_config.to_yaml(yaml_path)
