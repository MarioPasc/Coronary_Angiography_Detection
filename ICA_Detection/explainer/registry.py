"""
Simple global registry so the user only has to pass a *string* model name.

Example
-------
from ICA_Detection.explainer import build_explainer
cam = build_explainer("ultralytics", weight="runs/train/exp/weights/best.pt")
img  = cam("test.jpg")[0]
img.show()
"""

from typing import Dict, Type

_ADAPTERS: Dict[str, Type] = {}


def register_adapter(name: str):
    """Decorator – use on every Adapter subclass."""
    def _decorator(cls):
        name_lc = name.lower()
        if name_lc in _ADAPTERS:
            raise KeyError(f"Adapter name '{name}' already registered.")
        _ADAPTERS[name_lc] = cls
        return cls
    return _decorator


def get_adapter_cls(name: str):
    try:
        return _ADAPTERS[name.lower()]
    except KeyError as e:   # pragma: no cover
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(_ADAPTERS.keys())}"
        ) from e


def build_explainer(
    model_name: str,
    weight: str,
    **kwargs,
):
    """
    Factory that returns a `GradCAMExplainer` already wired to the chosen model.

    Any extra **kwargs are forwarded to GradCAMExplainer.
    """
    from ICA_Detection.explainer.gradcam import GradCAMExplainer  # local import to avoid cycles

    adapter_cls = get_adapter_cls(model_name)
    from ICA_Detection.explainer import LOGGER

    LOGGER.info(
        f"Building GradCAMExplainer for model '{model_name}' "
        f"using adapter '{adapter_cls.__name__}' with weight '{weight}'"
    )
    return GradCAMExplainer(adapter_cls=adapter_cls, weight=weight, **kwargs)
