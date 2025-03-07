# ICA_Detection/external/torchvision/detection

The original code for this module was taken from [torchvision reference module for detection](https://github.com/pytorch/vision/tree/main/references/detection). Several modifications have been made to some of these functions in order to implement additional functionality. 

- Add DEBUG global flag at the beginning of some scripts to enable debug prints
- Add saving images and predicted bboxes to a JSON file in `engine.py` at `evaluator`