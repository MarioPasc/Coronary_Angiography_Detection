import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


def freeze_vgg_layers(vgg_features, freeze_until=10):
    """
    Freeze the first N layers of the VGG feature extractor.
    freeze_until: number of layers (convs, ReLUs) to freeze from the start.
    By default, we freeze up to layer10.
    """
    count = 0
    for layer in vgg_features.children():
        for param in layer.parameters():
            if count < freeze_until:
                param.requires_grad = False
        count += 1


def get_ssd_model(pretrained=True, num_classes=2, freeze_until=10):
    """
    Creates and returns an SSD (Single Shot MultiBox Detector) model for lesion detection.

    Args:
        pretrained (bool): Whether to use an SSD pre-trained on COCO (SSD300 with VGG16).
        num_classes (int): Number of classes (including background).
        freeze_until (int): Number of layers to freeze in the VGG backbone.

    Returns:
        SSD: The model ready for training/inference.
    """
    if pretrained:
        # ------------------------------------------------
        # Use a pre-trained SSD (VGG16 backbone) from Torchvision
        # ------------------------------------------------
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

        # Optionally freeze some early VGG layers
        # The backbone is accessible via model.backbone, specifically model.backbone.body
        if hasattr(model.backbone, "body"):
            freeze_vgg_layers(model.backbone.body, freeze_until)

        # Replace the classification head for our number of classes
        in_channels = model.head.classification_head.in_channels
        num_anchors = model.head.classification_head.num_anchors

        model.head.classification_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Re-initialize regression head for consistency (optional)
        model.head.regression_head.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, padding=1
        )

    else:
        # ------------------------------------------------
        # Build SSD from scratch (VGG16 base, no COCO pretraining)
        # ------------------------------------------------

        # 1) Load a VGG16 backbone with NO pretrained weights
        vgg_backbone = torchvision.models.vgg16(pretrained=False).features

        # 2) Freeze some of the early layers if desired
        freeze_vgg_layers(vgg_backbone, freeze_until)

        # 3) Modify the backbone for SSD
        #    E.g. often you do: pool5 => ceil_mode = True, slice backbone to conv5_3
        if hasattr(vgg_backbone[16], "ceil_mode"):
            vgg_backbone[16].ceil_mode = True  # pool5

        # We'll slice up to index 23 to stop after conv5_3
        vgg_backbone = vgg_backbone[:23]

        # 4) Define the extra SSD layers
        extras = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # 5) The channels for each feature map
        backbone_out_channels = [512, 1024, 512, 256, 256, 256]

        # 6) Classification and Regression heads
        classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=backbone_out_channels,
            num_anchors=[4, 6, 6, 6, 4, 4],
            num_classes=num_classes,
        )
        regression_head = torchvision.models.detection.ssd.SSDRegressionHead(
            in_channels=backbone_out_channels,
            num_anchors=[4, 6, 6, 6, 4, 4],
        )

        # 7) Default Box (Anchor) Generator
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            min_ratio=0.2,
            max_ratio=0.95,
        )

        # 8) Build the SSD model
        model = SSD(
            backbone=vgg_backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            size=(300, 300),
            head=torchvision.models.detection.ssd.SSDHead(
                classification_head=classification_head, regression_head=regression_head
            ),
            backbone_out_channels=backbone_out_channels,
            extras=extras,
        )

    return model
