from ICA_Detection.external.ultralytics.ultralytics import YOLO
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")  # or use your model path

# Path to images and masks - update these to your actual paths
images_folder = (
    "/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/images"
)
masks_folder = (
    "/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/labels/masks"
)

# Select an image to process
image_filename = "arcadetest_p1_v1_00001.png"  # Replace with your actual image filename
image_path = os.path.join(images_folder, image_filename)

# Find the corresponding mask (assuming same base filename but possibly different extension)
image_basename = Path(image_filename).stem
mask_path = None

# Search for mask with matching basename
for mask_file in os.listdir(masks_folder):
    mask_basename = Path(mask_file).stem
    if mask_basename == image_basename:
        mask_path = os.path.join(masks_folder, mask_file)
        break

if not mask_path:
    raise ValueError(
        f"No corresponding mask found for {image_filename} in {masks_folder}"
    )

print(f"Processing image: {image_path}")
print(f"Using mask: {mask_path}")

# Load image and mask
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")  # Load as grayscale

# Preprocess image for YOLO
transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]
)
img = transform(image).unsqueeze(0)  # Add batch dimension

# Preprocess mask to match input size
mask_tensor = transforms.ToTensor()(
    transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.NEAREST)(
        mask
    )
).unsqueeze(
    0
)  # Shape: [1, 1, 640, 640]

# Dictionary to store intermediate feature maps
feature_maps = {}
masked_feature_maps = {}


# Function to hook and save feature maps
def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.clone()  # Store original feature map

        # Resize mask to match feature map dimensions using nearest neighbor
        feature_h, feature_w = output.shape[2], output.shape[3]
        resized_mask = transforms.Resize(
            (feature_h, feature_w), interpolation=transforms.InterpolationMode.NEAREST
        )(mask)
        resized_mask_tensor = transforms.ToTensor()(resized_mask).to(output.device)

        # Expand mask dimensions to match output channels
        expanded_mask = resized_mask_tensor.expand(
            1, output.shape[1], feature_h, feature_w
        )

        # Apply mask to feature map
        masked_output = output * expanded_mask
        masked_feature_maps[name] = masked_output

        # Return the masked output to modify what gets passed to the next layer
        return masked_output

    return hook


# Register hooks to capture and modify backbone outputs
hooks = []
for name, module in model.model.named_modules():
    # The backbone outputs are fed into C2f blocks in the neck
    if isinstance(module, torch.nn.Module) and name.startswith("model."):
        if (
            name == "model.15" or name == "model.18" or name == "model.21"
        ):  # P3, P4, P5 features
            print(f"Registering hook for {name}")
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Capture neck outputs if needed
    if isinstance(module, torch.nn.Module) and "detect" in name and "cv" in name:
        hooks.append(module.register_forward_hook(hook_fn(f"head_{name}")))

# Run inference with the modified feature maps
with torch.no_grad():
    results = model(img)

# Remove hooks after use
for hook in hooks:
    hook.remove()

# Print information about original and masked feature maps
print("\n=== Multi-scale Features Information ===")
for name in feature_maps.keys():
    original = feature_maps[name]
    masked = masked_feature_maps[name]

    print(f"\nLayer: {name}")
    print(f"  Original shape: {original.shape}")
    print(f"  Masked shape: {masked.shape}")

    # Calculate spatial reduction from input
    reduction = 640 // original.shape[-1]
    print(f"  Spatial reduction: {reduction}x")
    print(f"  Channels: {original.shape[1]}")
    print(f"  Spatial dimensions: {original.shape[2]}x{original.shape[3]}")

    # Optional: Visualize a sample channel from the original and masked feature maps
    channel_idx = 0  # Visualize first channel
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Original Feature Map - {name}")
    plt.imshow(original[0, channel_idx].cpu().numpy(), cmap="viridis")

    plt.subplot(1, 3, 2)
    plt.title(f"Mask (Resized to {original.shape[2]}x{original.shape[3]})")
    # Get the resized mask for visualization
    resized_mask = transforms.Resize(
        (original.shape[2], original.shape[3]),
        interpolation=transforms.InterpolationMode.NEAREST,
    )(mask)
    plt.imshow(np.array(resized_mask), cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title(f"Masked Feature Map - {name}")
    plt.imshow(masked[0, channel_idx].cpu().numpy(), cmap="viridis")

    plt.tight_layout()
    plt.savefig(f"feature_map_{name.replace('.', '_')}.png")

print("\n=== Summary ===")
print(f"Image: {image_filename}")
print(f"Mask: {os.path.basename(mask_path)}")
print(f"Total feature maps processed: {len(feature_maps)}")
