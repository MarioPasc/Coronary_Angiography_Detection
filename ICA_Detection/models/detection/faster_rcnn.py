import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def freeze_resnet_layers(backbone, freeze_until=2):
    """
    Freezes the specified number of layers in a ResNet backbone.
    freeze_until: an integer from 0 to 5, where
        0 => freeze none,
        1 => freeze conv1 and bn1,
        2 => freeze layer1,
        3 => freeze layer2,
        4 => freeze layer3,
        5 => freeze layer4
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


def get_faster_rcnn_model(pretrained=True, num_classes=2, freeze_until=2):
    """
    Creates and returns a Faster R-CNN model for lesion detection.

    Args:
        pretrained (bool): Whether to use a model pre-trained on the COCO dataset
                           (FasterRCNN + ResNet50 FPN).
        num_classes (int): Number of classes (including background).
        freeze_until (int): How many layers of the ResNet backbone to freeze (0 to 5).

    Returns:
        FasterRCNN: The model ready for training or inference.
    """
    if pretrained:
        # ------------------------------------------------
        # Use a pretrained Faster R-CNN w/ ResNet50 FPN
        # ------------------------------------------------
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Freeze part of the backbone if desired
        freeze_resnet_layers(model.backbone.body, freeze_until)

        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )
    else:
        # ------------------------------------------------
        # Build a Faster R-CNN from scratch (no COCO pretraining)
        # ------------------------------------------------

        # 1) Create a ResNet50 backbone with no pretrained weights
        backbone = torchvision.models.resnet50(pretrained=False)

        # 2) Optionally freeze certain early layers
        freeze_resnet_layers(backbone, freeze_until)

        # 3) Convert it to a feature extractor by removing the classifier layers
        #    We only need feature maps up to layer4.
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone_out_channels = 2048  # ResNet50 layer4 output channels

        # 4) Define an AnchorGenerator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        # 5) Define the ROI pooler:
        #    Since there's no FPN, we only have a single feature map
        #    so we can call it "0".
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],  # single feature map from the backbone
            output_size=7,
            sampling_ratio=2,
        )

        # 6) Build the model
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    return model
