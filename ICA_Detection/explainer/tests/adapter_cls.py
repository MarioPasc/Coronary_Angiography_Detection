from ICA_Detection.explainer.registry import get_adapter_cls
print(get_adapter_cls("ultralytics"))   # → <class 'UltralyticsAdapter'>
print(get_adapter_cls("dca_yolov8"))    # → <class 'DCAYOLOv8Adapter'>
