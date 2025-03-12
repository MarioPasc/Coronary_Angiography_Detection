import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, nms
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import _validate_trainable_layers


def freeze_backbone_layers(backbone, freeze_until=2):
    """
    Freezes the specified number of layers in a backbone.
    freeze_until: an integer from 0 to 5, where
        0 => freeze none
        1 => freeze initial layers (conv1/bn1)
        2 => also freeze layer1
        3 => also freeze layer2
        4 => also freeze layer3
        5 => also freeze layer4 (freeze all)
    """
    # For ResNet backbones
    if hasattr(backbone, "layer1"):
        for name, param in backbone.named_parameters():
            # Freeze conv1/bn1
            if freeze_until >= 1 and (
                name.startswith("conv1") or name.startswith("bn1")
            ):
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
    # For MobileNet, EfficientNet, or other backbones
    # Implemented as a simple proportion-based freezing
    else:
        all_params = list(backbone.named_parameters())
        total_layers = len(all_params)

        if freeze_until > 0:
            # Calculate how many layers to freeze based on freeze_until (1-5)
            # as a proportion of total layers
            freeze_count = int((freeze_until / 5) * total_layers)

            # Freeze the first freeze_count layers
            for i, (name, param) in enumerate(all_params):
                if i < freeze_count:
                    param.requires_grad = False


def get_backbone_model(backbone_name, pretrained=True):
    """
    Creates and returns a backbone model based on the name.

    Args:
        backbone_name (str): Name of the backbone (resnet50, resnet101, mobilenet_v2, etc.)
        pretrained (bool): Whether to use pretrained weights

    Returns:
        model: The backbone model
        out_channels_list: List of output channels for each layer
    """
    if backbone_name == "resnet50":
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        out_channels_list = [256, 512, 1024, 2048]
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

    elif backbone_name == "resnet101":
        backbone = torchvision.models.resnet101(pretrained=pretrained)
        out_channels_list = [256, 512, 1024, 2048]
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

    elif backbone_name == "mobilenet_v2":
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
        # Remove the classifier
        backbone = backbone.features
        # MobileNetV2 structure differs from ResNet
        out_channels_list = [24, 32, 96, 1280]  # Approximate values for key layers
        # These indices correspond roughly to the end of each "inverted residual" block group
        return_layers = {"3": "0", "6": "1", "13": "2", "18": "3"}

    elif backbone_name == "efficientnet_b0":
        backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
        # Remove the classifier
        backbone = backbone.features
        # EfficientNet structure
        out_channels_list = [24, 40, 112, 1280]  # Approximate values for key layers
        # These indices target the end of the MBConv blocks
        return_layers = {"2": "0", "3": "1", "5": "2", "8": "3"}

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Remove unnecessary layers (GAP, FC, etc.)
    if backbone_name.startswith("resnet"):
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)

    return backbone, out_channels_list, return_layers


def get_retina_net_model(
    pretrained=True,
    num_classes=2,
    freeze_until=2,
    backbone_name="resnet50",
    fpn_channels=256,
    anchor_sizes=None,
    anchor_aspect_ratios=None,
    nms_threshold=0.5,
    score_threshold=0.05,
):
    """
    Creates and returns a RetinaNet model with customizable hyperparameters.

    Args:
        pretrained (bool): Whether to start with a model pre-trained on COCO (RetinaNet+backbone)
        num_classes (int): Number of classes (including background)
        freeze_until (int): How many layers of the backbone to freeze (0 to 5)
        backbone_name (str): Name of the backbone to use (resnet50, resnet101, mobilenet_v2, efficientnet_b0)
        fpn_channels (int): Number of channels in the FPN
        anchor_sizes (list): List of anchor sizes (in pixels), default: [32, 64, 128, 256, 512]
        anchor_aspect_ratios (list): List of anchor aspect ratios, default: [0.5, 1.0, 2.0]
        nms_threshold (float): IoU threshold for NMS, default: 0.5
        score_threshold (float): Score threshold for detections, default: 0.05

    Returns:
        RetinaNet: The model ready for training or inference
    """
    # Set default values if not provided
    if anchor_sizes is None:
        anchor_sizes = [32, 64, 128, 256, 512]

    if anchor_aspect_ratios is None:
        anchor_aspect_ratios = [0.5, 1.0, 2.0]

    if pretrained and backbone_name == "resnet50":
        # Use the pre-trained RetinaNet with ResNet50 backbone
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights="DEFAULT",
            trainable_backbone_layers=5
            - freeze_until,  # Convert freeze_until to trainable layers
        )

        # Apply custom NMS threshold
        model.nms_thresh = nms_threshold
        model.score_thresh = score_threshold

        # Freeze backbone layers according to freeze_until
        freeze_backbone_layers(model.backbone.body, freeze_until)

        # Update anchor generator if custom sizes or aspect ratios are provided
        if anchor_sizes != [32, 64, 128, 256, 512] or anchor_aspect_ratios != [
            0.5,
            1.0,
            2.0,
        ]:
            anchor_generator = AnchorGenerator(
                sizes=tuple((s,) for s in anchor_sizes),
                aspect_ratios=tuple((ar,) for ar in anchor_aspect_ratios),
            )
            model.anchor_generator = anchor_generator

        # Modify FPN channels if different from default
        if fpn_channels != 256:
            # This requires modifying internal FPN components, which is complex
            # We'll keep the original FPN in this case, but note the requested change
            print(
                f"Warning: Changing FPN channels for pretrained model not implemented. Using default (256)."
            )

        # Replace the classification head to handle custom number of classes
        in_features = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors

        # Replace the 4-block conv layers
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

        # Replace the final classification conv
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_features,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Update num_classes attribute
        model.head.classification_head.num_classes = num_classes

    else:
        # Build a RetinaNet model with custom backbone
        # 1) Get the backbone model
        backbone, out_channels_list, return_layers = get_backbone_model(
            backbone_name, pretrained=pretrained
        )

        # 2) Freeze layers as requested
        freeze_backbone_layers(backbone, freeze_until)

        # 3) Create a Feature Pyramid Network
        backbone_with_fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            backbone,
            return_layers=return_layers,
            in_channels_list=out_channels_list,
            out_channels=fpn_channels,
            extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelP6P7(
                fpn_channels, fpn_channels
            ),
        )

        # 4) Define anchor generator with custom sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in anchor_sizes),
            aspect_ratios=tuple((ar,) for ar in anchor_aspect_ratios),
        )

        # 5) Build the RetinaNet model
        model = RetinaNet(
            backbone=backbone_with_fpn,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            score_thresh=score_threshold,
            nms_thresh=nms_threshold,
        )

    return model
