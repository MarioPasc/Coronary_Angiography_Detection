# Explainer module: GradCAM with custom YOLO implementations.

## Usage

Easy run:
```python
from ICA_Detection.explainer import build_explainer, display_images

# one line: choose model by name, point to weights
cam = build_explainer(
    "ultralytics",
    weight="runs/train/exp/weights/best.pt",
    method="GradCAM++",
    conf_threshold=0.3,
)

imgs = cam("sample.jpg")          # or a directory
display_images(imgs)
```

To further expland the module with more inspectors for new custom models:
```python
# explainer/adapters/my_yolo.py
from ICA_Detection.explainer.registry import register_adapter
from ICA_Detection.explainer.adapters.base import BaseAdapter

@register_adapter("my_yolo")
class MyAdapter(BaseAdapter):
    def _load_model(self, weight, device): ...
    def _get_class_names(self): ...
    def get_target_layers(self): ...
    def postprocess(self, preds): ...
```

## Credits

This module has been heavily inspired by the work of Sarma Borah, Proyash Paban and Kashyap, Devraj and Laskar, Ruhini Aktar and Sarmah, and Ankur Jyoti.
Please, if you use this module, do not forget to give proper credit to the original authors:

```bibtex
@ARTICLE{Sarma_Borah2024-un,
  title     = "A comprehensive study on Explainable {AI} using {YOLO} and post
               hoc method on medical diagnosis",
  author    = "Sarma Borah, Proyash Paban and Kashyap, Devraj and Laskar,
               Ruhini Aktar and Sarmah, Ankur Jyoti",
  journal   = "J. Phys. Conf. Ser.",
  publisher = "IOP Publishing",
  volume    =  2919,
  number    =  1,
  pages     = "012045",
  month     =  dec,
  year      =  2024,
  copyright = "https://creativecommons.org/licenses/by/4.0/"
}
```