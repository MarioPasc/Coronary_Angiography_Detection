import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork


def freeze_resnet_layers(backbone, freeze_until=2):
    """
    Freezes the specified number of layers in a ResNet backbone.
    freeze_until: an integer from 0 to 5, where
        0 => freeze none
        1 => freeze conv1 and bn1
        2 => also freeze layer1
        3 => also freeze layer2
        4 => also freeze layer3
        5 => also freeze layer4
    """
    for name, param in backbone.named_parameters():
        # Freeze conv1/bn1
        if freeze_until >= 1 and (name.startswith("conv1") or name.startswith("bn1")):
            param.requires_grad = False
        # Freeze layer1
        elif freeze_until >= 2 and name.startswith("layer1"):
            param.requires_grad = False
        # Freeze layer2
        elif freeze_until >= 3 and name.startswith("layer2"):
            param.requires_grad = False
        # Freeze layer3
        elif freeze_until >= 4 and name.startswith("layer3"):
            param.requires_grad = False
        # Freeze layer4
        elif freeze_until >= 5 and name.startswith("layer4"):
            param.requires_grad = False


def get_retina_net_model(pretrained=True, num_classes=2, freeze_until=2):
    """
    Creates and returns a RetinaNet model for lesion detection.

    Args:
        pretrained (bool): Whether to start with a model pre-trained on the COCO dataset
                           (RetinaNet + ResNet50 backbone).
        num_classes (int): Number of classes (including background).
        freeze_until (int): How many layers of the ResNet backbone to freeze (0 to 5).

    Returns:
        RetinaNet: The model ready for training or inference.
    """
    if pretrained:
        # ------------------------------------------------
        # Load a RetinaNet model already trained on COCO
        # ------------------------------------------------
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")

        # Optionally freeze part of the ResNet backbone
        # The backbone is accessible through model.backbone.body
        freeze_resnet_layers(model.backbone.body, freeze_until)

        # ------------------------------------------------
        # Overwrite the classification head (conv + cls_logits)
        # to handle custom number of classes
        # ------------------------------------------------
        in_features = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors

        # (1) Replace the 4-block conv layers
        model.head.classification_head.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # (2) Replace the final classification conv
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_features,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # (3) Update num_classes attribute
        model.head.classification_head.num_classes = num_classes

    else:
        # ------------------------------------------------
        # Build a RetinaNet model from scratch (no COCO pretraining)
        # ------------------------------------------------

        # 1) Build a ResNet50 backbone with or without ImageNet weights
        #    (If truly from scratch: pretrained=False)
        backbone = torchvision.models.resnet50(pretrained=False)

        # 2) Optionally freeze some layers
        freeze_resnet_layers(backbone, freeze_until)

        # 3) Remove the last layers of ResNet (we only need feature maps)
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)

        # 4) The number of output channels in layer4 of ResNet50 is 2048
        backbone_out_channels = 2048

        # 5) Create a Feature Pyramid Network (FPN)
        #    We'll let BackboneWithFPN handle it, rather than building
        #    a separate FPN ourselves.
        backbone_with_fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            backbone,
            return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        # 6) Define an anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),  # anchor sizes
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        # 7) Build the RetinaNet model
        model = RetinaNet(
            backbone=backbone_with_fpn,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
        )

    return model
