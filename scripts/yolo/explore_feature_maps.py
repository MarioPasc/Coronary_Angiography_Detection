from ICA_Detection.external.ultralytics.ultralytics import YOLO
import torch
import numpy as np

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")  # or use your model path

# Dictionary to store intermediate feature maps
feature_maps = {}


# Function to hook and save feature maps
def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output

    return hook


# Register hooks to capture backbone outputs
for name, module in model.model.named_modules():
    # The backbone outputs are fed into C2f blocks in the neck
    if isinstance(module, torch.nn.Module) and name.startswith("model."):
        print(f"Registering hook for {name}")
        if ".15" in name or ".18" in name or ".21" in name:  # P3, P4, P5 features
            module.register_forward_hook(hook_fn(name))

    # Capture neck outputs
    if isinstance(module, torch.nn.Module) and "detect" in name and "cv" in name:
        module.register_forward_hook(hook_fn(f"head_{name}"))

# Create a sample input (adjust size as needed)
img = torch.randn(1, 3, 640, 640)
# Normalize the input to 0-1
img = img / 255.0

# Run inference
with torch.no_grad():
    model(img)

# Print information about backbone and neck feature maps
print("\n=== Multi-scale Features Information ===")
for name, feature in feature_maps.items():
    print(f"\nLayer: {name}")
    if isinstance(feature, (list, tuple)):
        for i, f in enumerate(feature):
            print(f"  Feature map {i}: shape={f.shape}")
    else:
        print(f"  Shape: {feature.shape}")
        # print(f"  Type: {feature.dtype}")
        # Calculate spatial reduction from input
        reduction = 640 // feature.shape[-1]
        # print(f"  Spatial reduction: {reduction}x")
        print(f"  Channels: {feature.shape[1]}")
        print(f"  Spatial dimensions: {feature.shape[2]}x{feature.shape[3]}")

print("\n=== Summary of Multi-scale Features ===")
print(f"Total number of captured feature maps: {len(feature_maps)}")
