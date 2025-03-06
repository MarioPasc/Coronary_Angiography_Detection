import torch
import torchvision
from torchvision.models.detection import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


def get_ssd_model(pretrained=True, num_classes=2):  # 2 classes: background and lesion
    """
    Creates and returns an SSD (Single Shot MultiBox Detector) model for lesion detection.

    Args:
        pretrained (bool): Whether to use a model pre-trained on COCO dataset
        num_classes (int): Number of classes (including background)

    Returns:
        SSD: The model ready for training/inference
    """
    if pretrained:
        # Start with a pre-trained model (SSD300 with VGG16 backbone)
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

        # Replace the classifier head for our number of classes
        in_channels = model.head.classification_head.in_channels
        num_anchors = model.head.classification_head.num_anchors

        # Create new classification head
        model.head.classification_head.cls_logits = torch.nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )
    else:
        # Start with a model from scratch using VGG16 backbone
        backbone = torchvision.models.vgg16(pretrained=True).features

        # Modify backbone as needed for SSD
        # Extract specific feature maps from VGG that SSD uses
        backbone_out_channels = 512

        # Define anchor (default box) generator
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            min_ratio=0.2,
            max_ratio=0.95,
        )

        # Create the model
        model = SSD(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            image_size=(300, 300),  # Standard SSD input size
        )

    return model
