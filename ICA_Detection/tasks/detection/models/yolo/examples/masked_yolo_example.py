import numpy as np
from pathlib import Path
from ICA_Detection.tasks.detection.models.yolo.masked_yolo_model import MaskedYOLOModel

def create_sample_npz(output_path, image_path):
    """
    Create a sample .npz file with an image and mask for testing.
    
    Args:
        output_path (str): Path to save the .npz file
        image_path (str): Path to the source image
    """
    from PIL import Image
    import numpy as np
    
    # Load image
    img = np.array(Image.open(image_path))
    
    # Create a simple mask (bright center, dark edges)
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-((x - w/2)**2 + (y - h/2)**2) / (w/4)**2)
    mask = (mask * 255).astype(np.uint8)
    
    # Save as .npz
    np.savez(output_path, image=img, mask=mask)
    print(f"Created sample .npz file at {output_path}")

def main():
    # Create a sample .npz file
    sample_img = "path/to/sample/image.jpg"  # Replace with actual image path
    sample_npz = "sample_with_mask.npz"
    
    if not Path(sample_npz).exists():
        create_sample_npz(sample_npz, sample_img)
    
    # Initialize the masked YOLO model
    model = MaskedYOLOModel("yolov8n.pt")
    
    # Run prediction with the .npz file
    results = model.predict(sample_npz)
    
    # Process and display results
    for r in results:
        print(f"Found {len(r.boxes)} objects")
        print(f"Classes: {r.boxes.cls}")
        print(f"Confidence: {r.boxes.conf}")
    
    # Save results visualization
    for i, r in enumerate(results):
        r.save(filename=f"result_{i}.jpg")

if __name__ == "__main__":
    main()
