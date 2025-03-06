import torch
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork


def get_retina_net_model(
    pretrained=True, num_classes=2
):  # 2 classes: background and lesion
    """
    Creates and returns a RetinaNet model for lesion detection.

    Args:
        pretrained (bool): Whether to use a model pre-trained on COCO dataset
        num_classes (int): Number of classes (including background)

    Returns:
        RetinaNet: The model ready for training/inference
    """
    # Load a pre-trained model
    if pretrained:
        # Start with a pre-trained model
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

        # Replace the classifier with a new one for our number of classes
        in_features = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors

        # Create new classification head
        model.head.classification_head.num_classes = num_classes
        model.head.classification_head.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_features, in_features, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_features, in_features, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_features, in_features, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_features, in_features, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_features,
                num_anchors * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
    else:
        # Start with a model from scratch
        # Load backbone
        backbone = torchvision.models.resnet50(pretrained=True)

        # Remove the last layers as we don't need them
        modules = list(backbone.children())[:-2]
        backbone = torch.nn.Sequential(*modules)

        # FPN needs to know the number of output channels in the backbone
        backbone_out_channels = 2048

        # Define Feature Pyramid Network
        fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048], out_channels=256
        )

        # Combine backbone and FPN
        backbone_with_fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            backbone,
            return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        # Define anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),  # Different sizes of anchors
            aspect_ratios=((0.5, 1.0, 2.0),),  # Different aspect ratios
        )

        # Create the model
        model = RetinaNet(
            backbone=backbone_with_fpn,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
        )

    return model
