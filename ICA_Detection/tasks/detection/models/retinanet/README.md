# RetinaNet model

The majority of this code has been taken from user yhenon in github, from [this repository](https://github.com/yhenon/pytorch-retinanet). The code provies a comprehensive implementation of the RetinaNet model, including the ResNet, PyramidFeatures, and the Regression and Classification head. Yhenon's RetinaNet archieved similar accuracy in the COCO dataset as the original RetinaNet paper.

## Log changes

1. Changed th way `forward()` is called to accept the images and labels entry, in order to be consistent with the `engine.py` implementation from `torchvision/references/detection`.

```python
# previous: 
    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        ...
# now:
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided.")

        if self.training:
            img_batch, annotations = images, targets
        else:
            img_batch = images
        ...
```
