import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_mask_rcnn_model(
    pretrained=True, num_classes=2, return_masks=True
):  # 2 classes: background and lesion
    """
    Creates and returns a Mask R-CNN model for lesion detection and segmentation.

    Args:
        pretrained (bool): Whether to use a model pre-trained on COCO dataset
        num_classes (int): Number of classes (including background)
        return_masks (bool): Whether to return segmentation masks (can be disabled if only bounding boxes are needed)

    Returns:
        MaskRCNN: The model ready for training/inference
    """
    if pretrained:
        # Start with a pre-trained model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # Replace the box classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )

        # Replace the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = (
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )
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

        # Define anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),  # Different sizes of anchors
            aspect_ratios=((0.5, 1.0, 2.0),),  # Different aspect ratios
        )

        # Define ROI pooler for box head
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],  # Names of the feature maps from FPN
            output_size=7,
            sampling_ratio=2,
        )

        # Define ROI pooler for mask head
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
        )

        # Create the model
        model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler,
        )

    # Disable mask return if not needed
    if not return_masks:
        model.return_masks = False

    return model
