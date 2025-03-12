import torch
from torch import nn


class RetinaNetWrapper(nn.Module):
    def __init__(self, retinanet_model):
        super(RetinaNetWrapper, self).__init__()
        self.model = retinanet_model

    def forward(self, images, targets=None):
        """
        Wrapper to handle format conversion between engine and RetinaNet

        Args:
            images: Tensor or list of images
            targets: List of target tensors from the data loader (format from collate_fn)
                     Each tensor has shape [N, 5] where the columns are [x1, y1, x2, y2, class_id]

        Returns:
            During training: loss dict
            During inference: detection results
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided.")

        # If images is a list, stack them
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        # During training, convert target format for RetinaNet
        if self.training and targets is not None:
            # Convert list of tensors to a batched tensor [batch_size, max_objects, 5]
            batch_size = len(targets)
            max_objects = max(t.shape[0] for t in targets)

            # Create a padded tensor filled with -1 (to indicate padding)
            batched_targets = (
                torch.ones(
                    (batch_size, max_objects, 5),
                    dtype=targets[0].dtype,
                    device=targets[0].device,
                )
                * -1
            )

            # Fill in the actual values
            for i, target in enumerate(targets):
                if target.shape[0] > 0:  # If there are any objects
                    batched_targets[i, : target.shape[0], :] = target

            # Pass to the model
            return self.model(images, batched_targets)
        else:
            # For inference, just pass through
            return self.model(images)
