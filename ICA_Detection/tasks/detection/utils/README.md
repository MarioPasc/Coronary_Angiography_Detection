# ICA_Detection/tasks/utils

The original code for this module was taken from [torchvision reference module for detection](https://github.com/pytorch/vision/tree/main/references/detection). Several modifications have been made to some of these functions in order to implement additional functionality.

- Add DEBUG global flag at the beginning of some scripts to enable debug prints
- Add saving images and predicted bboxes to a JSON file in `engine.py` at `evaluator`
- Add `early_stopping.py` script, which implement a simple early stopping loss-based mechanism, as well as a best checkpoint saver and loader.
- Add `compute_validation_loss` at `engine.py`. Because detection models return losses only when in training mode (they return predictions otherwise), we briefly set the model to train()—but still wrap it in torch.no_grad() so no gradients are computed. The function accumulates the total loss over the validation set, then returns the average.
