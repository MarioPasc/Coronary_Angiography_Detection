# Coronary_Angiography_Detection

Models are in models/ folder.
There is an example script to perform evaluation with a YOLO model in inference/, along with some example images.
```
Coronary_Angiography_Detection
├─ ICA_Detection
│  ├─ __init__.py
│  ├─ external
│  ├─ generator
│  │  ├─ __init__.py
│  │  └─ generator.py
│  ├─ integration
│  │  ├─ README.md
│  │  ├─ __init__.py
│  │  ├─ arcade.py
│  │  ├─ cadica.py
│  │  ├─ integrate.py
│  │  └─ kemerovo.py
│  ├─ models
│  │  ├─ __init__.py
│  │  └─ detection
│  │     ├─ __init__.py
│  │     ├─ faster_rcnn.py
│  │     ├─ retinanet.py
│  │     ├─ train.py
│  │     └─ yolo.py
│  ├─ preprocessing
│  │  ├─ README.md
│  │  ├─ __init__.py
│  │  ├─ planner.py
│  │  └─ preprocessing.py
│  ├─ splits
│  │  ├─ __init__.py
│  │  ├─ holdout_pytorch_models.py
│  │  └─ holdout_yolo.py
│  └─ tools
│     ├─ README.md
│     ├─ __init__.py
│     ├─ bbox_translation.py
│     ├─ clahe.py
│     ├─ connected_components.py
│     ├─ dataset_conversions.py
│     ├─ dtype_standarization.py
│     ├─ format_standarization.py
│     ├─ fse.py
│     ├─ highpass.py
│     ├─ histogram_segmentation.py
│     ├─ lowpass.py
│     └─ resolution.py
├─ LICENSE
├─ README.md
├─ docs
│  └─ TODO.md
├─ pyproject.toml
└─ scripts
   ├─ generate_dataset.py
   └─ preproc_performance.py

```